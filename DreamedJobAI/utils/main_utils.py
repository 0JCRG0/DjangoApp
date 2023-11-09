import os
import openai
import psycopg2
from psycopg2.extensions import cursor
import numpy as np
import pandas as pd
from scipy import spatial
import pretty_errors
import timeit
import logging
import time
import asyncio
import asyncio
from openai import AsyncOpenAI
from openai import APIError, APIConnectionError, Timeout, RateLimitError
import json
from dotenv import load_dotenv
import logging
from aiohttp import ClientSession
from typing import Tuple
import re
import tiktoken
from typing import Callable
import pandas as pd
from datetime import datetime, timedelta
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pyarrow.parquet as pq
from aiohttp import ClientSession
import os

load_dotenv('.env')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")
SAVE_PATH = os.getenv("SAVE_PATH")
E5_BASE_V2_DATA = os.getenv("E5_BASE_V2_DATA")
COUNTRIES_JSON_DATA = os.getenv("COUNTRIES_JSON_DATA")
LOGGER_DJANGO = os.getenv("LOGGER_DJANGO")
LOGGER_DIR_PATH = os.getenv("LOGGER_DIR_PATH")
MODEL= "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"


delimiters_summary = "----"
delimiters_job_info = '####'

system_prompt_summary = f""" 

Your task is to extract the specified information from a job opening/
posted by a company, with the aim of effectively matching /
potential candidates for the position./

The job opening below is delimited by {delimiters_summary} characters./
Within each job opening there are three sections delimited by {delimiters_job_info} characters: title, location and description./

Extract the following information from its respective section and output your response in the following format:/

Title: found in the "title" section.
Location: found in the "location" section or in the "description" section.
Job Objective: found in the "description" section.
Responsibilities/Key duties: found in the "description" section.
Qualifications/Requirements/Experience: found in the "description" section.
Preferred Skills/Nice to Have: found in the "description" section.
About the company: found in the "description" section.
Compensation and Benefits: found in the "description" section.

"""

#Specs from 8/11/23
specs_gpt4_models = {
	"gpt-4-1106-preview": (128000, 0.01, 0.03, "json_object"),
	"gpt-4-vision-preview": (128000, 0.01, 0.03, "json_object"),
	"gpt-4": (8192, 0.03, 0.06, "text"),
	"gpt-4-0613": (8192, 0.03, 0.06, "text"),
	"gpt-4-32k": (32768, 0.06, 0.12, "text"),
	"gpt-4-32k-0613	": (32768, 0.06, 0.12, "text")
}

def count_words(text: str) -> int:
	# Remove leading and trailing whitespaces
	text = text.strip()

	# Split the text into words using whitespace as a delimiter
	words = text.split()

	# Return the count of words
	return len(words)

async def async_summarise_job_gpt(job_description: str, gpt_model: str="gpt-3.5-turbo-1106") -> Tuple[str, float]:
	
	client = AsyncOpenAI(
		# defaults to os.environ.get("OPENAI_API_KEY")
		api_key=OPENAI_API_KEY,
	)
	
	response = await client.chat.completions.create(
		model=gpt_model,
		messages=[
			{'role': 'user', 'content': system_prompt_summary},
			{'role': 'user', 'content': f"Job Opening: {delimiters_summary}{job_description}{delimiters_summary}"},
		],
		
		temperature=0,
		max_tokens = 400
	)
	response_message = response.choices[0].message.content

	usage = dict(response).get('usage')
	
	cost_per_summary = 0
	prompt_tokens = usage.prompt_tokens
	completion_tokens = usage.completion_tokens

	cost_per_token = {
		"gpt-3.5-turbo-1106": (0.001, 0.002),
		"gpt-3.5-turbo": (0.0015, 0.002),
		"gpt-3.5-turbo-16k": (0.003, 0.004)
	}

	if gpt_model in cost_per_token:
		prompt_cost_per_k = cost_per_token[gpt_model][0]
		completion_cost_per_k = cost_per_token[gpt_model][1]
		prompt_cost = round((prompt_tokens / 1000) * prompt_cost_per_k, 3)
		completion_cost = round((completion_tokens / 1000) * completion_cost_per_k, 3)
		cost_per_summary = prompt_cost + completion_cost
		# logging.info(f"COST FOR SUMMARISING: ${total_cost:.4f} USD")
	else:
		logging.error("The gpt_model selected in invalid. Choose a valid option. See https://openai.com/blog/new-models-and-developer-products-announced-at-devday")
		raise Exception("The gpt_model selected in invalid. Choose a valid option. See https://openai.com/blog/new-models-and-developer-products-announced-at-devday")
	return response_message, cost_per_summary

async def async_summarise_description(description: str, gpt_model: str="gpt-3.5-turbo-1106") -> tuple:

	cost_per_summary = 0

	async def process_description(session, text):
		attempts = 0
		while attempts < 5:
			try:
				words_per_text = count_words(text)
				if words_per_text > 50:
					description_summary, cost_per_summary = await async_summarise_job_gpt(job_description=text, gpt_model=gpt_model)
					return description_summary, cost_per_summary
				else:
					logging.warning(f"Description is too short for being summarised. Number of words: {words_per_text}")
					return text, 0
			except (Exception) as e:
				attempts += 1
				print(f"{e}. Retrying attempt {attempts}...")
				logging.warning(f"{e}. Retrying attempt {attempts}...")
				await asyncio.sleep(5**attempts)  # exponential backoff
		else:
			print(f"Description could not be summarised after 5 attempts.")
			return text, 0

	async with ClientSession() as session:
		result = await process_description(session, description)

	cost_per_summary = result[1]


	return result[0], cost_per_summary

def num_tokens(text: str, model: str ="gpt-3.5-turbo-1106") -> int:
	#Return the number of tokens in a string.
	encoding = tiktoken.encoding_for_model(model)
	return len(encoding.encode(text))

def LoggingDjango():
	# Define a custom format with bold text
	log_format = '%(asctime)s %(levelname)s: \n%(message)s\n'

	# Configure the logger with the custom format
	logging.basicConfig(filename=LOGGER_DJANGO,
						level=logging.INFO,
						format=log_format)

def LoggingTest():
	# Define a custom format with bold text
	log_format = '%(asctime)s %(levelname)s: \n%(message)s\n'

	# Configure the logger with the custom format
	logging.basicConfig(filename=LOGGER_DIR_PATH + "/LoggingTest.log",
						level=logging.INFO,
						format=log_format)

def append_parquet(new_df: pd.DataFrame, filename: str):
	# Ensure user_id is int
	new_df['user_id'] = new_df['user_id'].astype(int)
	# Load existing data
	df = pd.read_parquet(SAVE_PATH + f'/{filename}.parquet')
	
	logging.info(f"Preexisting df: {df}")
	logging.info(f"df to append: {new_df}")

	df = pd.concat([df, new_df], ignore_index=True)
	df = df.drop_duplicates(subset='id', keep='last')

	# Write back to Parquet
	df.to_parquet(SAVE_PATH + f'/{filename}.parquet', engine='pyarrow')
	logging.info(f"{filename}.parquet has been updated")

def average_pool(last_hidden_states: Tensor,
				attention_mask: Tensor) -> Tensor:
	last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
	return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def e5_base_v2_query(user_cv: str) -> np.ndarray:
	tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
	model = AutoModel.from_pretrained('intfloat/e5-base-v2')

	query_e5_format = f"query: {user_cv}"

	batch_dict = tokenizer(query_e5_format, max_length=512, padding=True, truncation=True, return_tensors='pt')

	outputs = model(**batch_dict)
	query_embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).detach().numpy().flatten()
	return query_embedding

def filter_last_two_weeks(df:pd.DataFrame) -> pd.DataFrame:
	# Get the current date
	current_date = datetime.now().date()
	
	# Calculate the date two weeks ago from the current date
	two_weeks_ago = current_date - timedelta(days=14)
	
	# Filter the DataFrame to keep only rows with timestamps in the last two weeks
	filtered_df = df[df["timestamp"].dt.date >= two_weeks_ago]
	
	return filtered_df

def set_dataframe_display_options():
	# Call the function to set the desired display options
	pd.set_option('display.max_columns', None)  # Show all columns
	pd.set_option('display.max_rows', None)  # Show all rows
	pd.set_option('display.width', None)  # Disable column width restriction
	pd.set_option('display.expand_frame_repr', False)  # Disable wrapping to multiple lines
	pd.set_option('display.max_colwidth', None)  # Display full contents of each column

def filter_df_per_countries(df: pd.DataFrame, user_desired_country: str, user_second_desired_country: str) -> pd.DataFrame:
	# Load the JSON file into a Python dictionary
	with open(COUNTRIES_JSON_DATA, 'r') as f:
		data = json.load(f)

	# Function to get country information
	def get_country_info(user_country):
		values = []
		for continent, details in data.items():
			for country in details['Countries']:
				if country['country_name'] == user_country:
					values.append(country['country_name'])
					values.append(country['country_code'])
					values.append(country['capital_english'])
					subdivisions = country.get('subdivisions')  # Use get() to safely access the key
					if subdivisions:
						if isinstance(subdivisions, list):
							for subdivision in subdivisions:
								if isinstance(subdivision, dict) and 'subdivisions_code' in subdivision and 'subdivisions_name' in subdivision:
									values.append(subdivision['subdivisions_code'])
									values.append(subdivision['subdivisions_name'])
								else:
									pass
						else:
							pass
					else:
						pass
		return values

	# Get information for the first desired country
	country_values1 = get_country_info(user_desired_country)

	# Initialize country_values2 as an empty list
	country_values2 = []

	# Check if the second desired country is not empty
	if user_second_desired_country:
		country_values2 = get_country_info(user_second_desired_country)

	# Combine both sets of country values
	all_country_values = country_values1 + country_values2

	# Convert 'location' column to lowercase
	df['location'] = df['location'].str.lower()    

	# Convert all country values to lowercase and escape special characters
	country_values_lower = [re.escape(value.lower()) for value in all_country_values]
	
	# Create a mask with all False
	mask = pd.Series(False, index=df.index)

	# Update the mask if 'location' column contains any of the country values
	for value in country_values_lower:
		mask |= df['location'].str.contains(value, na=False)

	# Filter DataFrame
	filtered_df = df[mask]

	return filtered_df

def preexisting_ids_postgre(user_id:int) -> list :
	conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)

	# Create a cursor object
	cur = conn.cursor()

	# Fetch new data from the table where id is greater than max_id
	cur.execute(f"SELECT job_id FROM \"DreamedJobAI_suitablejobs\" WHERE user_id = {user_id}")
	
	data = cur.fetchall()

	cur.close()
	conn.close()
	
	# Separate the columns into individual lists
	job_ids = [row[0] for row in data]

	return job_ids

def find_unique_ids(ids: tuple, preexisting_ids: list) -> list:
	# Use a list comprehension to find the numbers in ids that are not in preexisting_ids
	unique_ids = [id for id in ids if id not in preexisting_ids]
	return unique_ids

def list_or_dict_json_output_GPT4(json_output_GPT4):
	if isinstance(json_output_GPT4, list):
		# If json_output_GPT4 is a list, it contains multiple records
		logging.info("json_output_GPT4 is a list of dictionaries, it contains multiple records")
		df_json_output_GPT4 = pd.read_json(json.dumps(json_output_GPT4))
		return df_json_output_GPT4	
	elif isinstance(json_output_GPT4, dict):
		# If json_output_GPT4 is a dictionary, it contains a single record
		logging.info("json_output_GPT4 is a dictionary, it contains a single record")
		data = [json_output_GPT4]
		df = pd.DataFrame(data)
		df['id'] = df['id'].astype(int)
		return df
	else:
		# Handle other cases if necessary
		pass

def fetch_all_country_values(user_desired_country: str, user_second_desired_country: str | None) -> list:
	# Load the JSON file into a Python dictionary
	with open(COUNTRIES_JSON_DATA, 'r') as f:
		data = json.load(f)

	# Function to get country information
	def get_country_info(user_country: str = user_desired_country):
		values = []
		for continent, details in data.items():
			for country in details['Countries']:
				if country['country_name'] == user_country:
					values.append(country['country_name'])
					values.append(country['country_code'])
					values.append(country['capital_english'])
					subdivisions = country.get('subdivisions')  # Use get() to safely access the key
					if subdivisions:
						if isinstance(subdivisions, list):
							for subdivision in subdivisions:
								if isinstance(subdivision, dict) and 'subdivisions_code' in subdivision and 'subdivisions_name' in subdivision:
									values.append(subdivision['subdivisions_code'])
									values.append(subdivision['subdivisions_name'])
								else:
									pass
						else:
							pass
					else:
						pass
		return values

	# Get information for the first desired country
	country_values1 = get_country_info()

	# Initialize country_values2 as an empty list
	country_values2 = []

	# Check if the second desired country is not empty
	if user_second_desired_country:
		country_values2 = get_country_info(user_second_desired_country)

	# Combine both sets of country values
	all_country_values = country_values1 + country_values2

	return all_country_values

def fetch_top_n_matching_jobs(
		user_cv_embedding: np.ndarray,
		all_country_values:str,
		cursor: cursor,
		top_n: str,
		similarity_or_distance_metric: str = "NN",
		table_name: str ="embeddings_e5_base_v2",
		interval_days: str = '\'15 days\''
	) -> pd.DataFrame:
	
	metric_mapping = {
		"NN": "<->",
		"inner_product": "<#>",
		"cosine": "<=>"
	}

	# Check if the provided value exists in the dictionary
	if similarity_or_distance_metric in metric_mapping:
		similarity_metric = metric_mapping[similarity_or_distance_metric]
	else:
		logging.error("""Invalid similarity_or_distance_metric. Choose "NN", "inner_product" or "cosine" """)
		raise Exception("""Invalid similarity_or_distance_metric. Choose "NN", "inner_product" or "cosine" """)

	country_values_str = "{" + ",".join(all_country_values).lower() + "}"

	query = f"""
    SELECT *
    FROM (
        SELECT *
        FROM {table_name}
        WHERE timestamp >= (current_date - interval {interval_days})
    ) AS two_weeks
    WHERE substring(lower(job_info) from '#### location: (.*?) ####') = ANY(%s::text[])
    ORDER BY embedding {similarity_metric} %s;
	"""
	cursor.execute(query.format(table_name="embeddings_e5_base_v2"), (country_values_str, user_cv_embedding))

	# Fetch all the rows
	rows = cursor.fetchall()

	# Separate the columns into individual lists
	ids = [row[0] for row in rows]
	jobs_info = [row[1] for row in rows]

	df = pd.DataFrame({'id': ids, 'job_info': jobs_info})

	return df


async def async_format_top_jobs_summarize(
	user_id: int,
	user_cv: str,
	df: pd.DataFrame,
	summarize_gpt_model: str,
	classify_gpt_model: str,
) -> Tuple[str, list[str], int]:
	

	specs_per_model = {
		"gpt-4-1106-preview": (128000, 0.01, 0.03),
		"gpt-4-vision-preview": (128000, 0.01, 0.03),
		"gpt-4": (8192, 0.03, 0.06),
		"gpt-4-32k": (32768, 0.06, 0.12),
		"gpt-3.5-turbo-1106": (16385, 0.001, 0.002),
		"gpt-3.5-turbo": (4096, 0.0015, 0.002),
		"gpt-3.5-turbo-16k": (16385, 0.003, 0.004)
	}

		
	if classify_gpt_model in specs_per_model:
		token_budget = specs_per_model[classify_gpt_model][0]
	
	ids = df['id'].tolist()
	if ids:
		#Basically giving the most relevant IDs from the previous function
		message = introduction_prompt

		start_time = asyncio.get_event_loop().time()

		tasks = [async_summarise_description(df[df['id'] == id]['job_info'].values[0], gpt_model=summarize_gpt_model) for id in ids]

		# Run the tasks concurrently
		results = await asyncio.gather(*tasks)
		job_summaries = []
		total_cost_summaries = 0    

		for id, result in zip(ids, results):
			job_description_summary, cost = result
			
			# Append summary to the list
			job_summaries.append({
				"id": id,
				"summary": job_description_summary,
				"user_id": user_id
			})

			#Append total cost
			total_cost_summaries += cost

			next_id = f'\nID:<{id}>\nJob Description:---{job_description_summary}---\n'
			if (
				num_tokens(message + next_id + user_cv, model=classify_gpt_model)
				> token_budget
			):
				break
			else:
				message += next_id
		
		logging.info(f"Total cost for summarising: ${total_cost_summaries} USD")

		elapsed_time = asyncio.get_event_loop().time() - start_time
		logging.info(f"\nElapsed time in async_format_top_jobs_summarize(). \nAll matching jobs summarised in: {elapsed_time:.2f} seconds")

		return message, job_summaries
	else:
		logging.error("DF does not have ids. Check fetch_top_n_matching_jobs().")
		#raise Exception("DF does not have ids. Check fetch_top_n_matching_jobs().")
		pass

async def async_classify_jobs_gpt_4(
	#This query is your question, only parameter to fill in function
	user_cv: str,
	formatted_message: str,
	classify_gpt_model: str = "gpt-4",
	log_gpt_messages: bool = True,
	specs_gpt4_models: dict=specs_gpt4_models,
):
	
	client = AsyncOpenAI(
		# defaults to os.environ.get("OPENAI_API_KEY")
		api_key=OPENAI_API_KEY,
	)

	start_time = asyncio.get_event_loop().time()

	if classify_gpt_model in specs_gpt4_models:
		input_cost = specs_gpt4_models[classify_gpt_model][1]
		output_cost = specs_gpt4_models[classify_gpt_model][2]
		response_format_enabled = specs_gpt4_models[classify_gpt_model][3]
	else:
		logging.error("The gpt_model selected in invalid. Choose a valid option. See https://openai.com/blog/new-models-and-developer-products-announced-at-devday")
		raise Exception("The gpt_model selected in invalid. Choose a valid option. See https://openai.com/blog/new-models-and-developer-products-announced-at-devday")
	
	response = await client.chat.completions.create(
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": f"{delimiters}{user_cv}{delimiters}"},
			{"role": "assistant", "content": formatted_message}
		],
		
		model=classify_gpt_model,
		temperature=0,
		max_tokens = 1000,
		#To use this you either need the newer gpt4 or turbo 1106
		response_format= { "type":f"{response_format_enabled}" }
	)
	response_message = response.choices[0].message.content
	
	if log_gpt_messages:
		logging.info(system_prompt + delimiters + user_cv + delimiters + formatted_message)

	usage = dict(response).get('usage')

	total_tokens = usage.total_tokens
	prompt_tokens = usage.prompt_tokens
	completion_tokens = usage.completion_tokens

	prompt_cost = round((prompt_tokens / 1000) * input_cost, 3)
	completion_cost = round((completion_tokens / 1000) * output_cost, 3)
	total_classifying_cost = prompt_cost + completion_cost
	logging.info(f"""\nUSING: "{classify_gpt_model}" FOR CLASSIFICATION \n\nINPUT:\nTOKENS USED: {prompt_tokens}.\nPROMP COST: ${prompt_cost}USD\n\nOUTPUT:\nTOKENS USED:{completion_tokens}\nCOMPLETION COST: {completion_cost}.\n\nTOTAL TOKENS USED:{total_tokens}\nTOTAL COST FOR CLASSIFYING: ${total_classifying_cost:.3f} USD""")

	elapsed_time = asyncio.get_event_loop().time() - start_time
	logging.info(f"""\n"{classify_gpt_model}" finished classifying! all in: {elapsed_time:.2f} seconds \n""")
	
	return response_message

def whether_json_object(gpt4_response: str) -> bool:
	try:
		json.loads(gpt4_response)
		logging.info(f"Response is a valid JSON object. Continuing...")
		return True
	except json.JSONDecodeError as e:
		logging.warning(f"JSON decoding error: {e}. Retrying async_classify_jobs_gpt_4()", exc_info=True)
		return False


async def retrying_async_classify_jobs_gpt_4(
		async_classify_jobs_gpt_4: Callable,
		user_cv: str,
		formatted_message: str,
		log_gpt_messages: bool,
		):
	
	default = '[{"id": "", "suitability": "", "explanation": ""}]'
	default_json = json.loads(default)

	#specs_gpt4_models is initialised in the beginning
	for model_name, model_specs in specs_gpt4_models.items():
		retries = 6  
		for i in range(retries):
			logging.info(f"""Using "{model_name}" for loop number: {i + 1}...""")
			try:
				gpt4_response = await async_classify_jobs_gpt_4(
											user_cv,
											formatted_message,
											model_name,
											log_gpt_messages
										)
				try:
					data = json.loads(gpt4_response)
					logging.info(f"""Response is a valid json object.\nModel used: "{model_name}"" Done in loop number: {i + 1}""")
					return data
				except json.JSONDecodeError:
					pass
			except openai.RateLimitError as e:
				logging.warning(f"{e}. Retrying in 10 seconds. Model: {model_name}, Number of retries: {i + 1}")
				time.sleep(10)
			except Exception as e:
				logging.warning(f"{e}. Retrying in 5 seconds. Model: {model_name}, Number of retries: {i + 1}", exc_info=True)
				time.sleep(5)

	logging.error("Check logs!!!! Main function was not callable. Setting json to default")
	return default_json

#Get the ids
def ids_df_most_suitable(df: pd.DataFrame) -> str:
	ids = ""
	for _, row in df.iterrows():
		if "id" in row:
			if ids:
				ids += ", "
			ids += f"'{row['id']}'"

	return f"({ids})"

def find_jobs_main_jobs_per_ids(cur: cursor, ids:str, table: str = "main_jobs") -> pd.DataFrame:
	#TABLE SHOULD EITHER BE "main_jobs" or "test"
	cur.execute( f"SELECT id, title, link, location, pubdate FROM {table} WHERE id IN {ids}")

	# Fetch all rows from the table
	rows = cur.fetchall()

	# Separate the columns into individual lists
	all_ids = [row[0] for row in rows]
	all_titles = [row[1] for row in rows]
	all_links = [row[2] for row in rows]
	all_locations = [row[3] for row in rows]
	all_pubdates = [row[4] for row in rows]

	df = pd.DataFrame({
		'id': all_ids,
		'title': all_titles,
		'link': all_links,
		'location': all_locations,
		'pubdate': all_pubdates
	})

	return df

def postgre_insert_matched_jobs(cursor: cursor, df: pd.DataFrame(), table_name: str = "matched_jobs"):

	start_time = timeit.default_timer()

	create_table = f"""
		CREATE TABLE IF NOT EXISTS {table_name} (
			id INTEGER UNIQUE,
			title TEXT,
			link TEXT,
			location TEXT,
			pubdate TIMESTAMP,
			summary TEXT,
			user_id INTEGER,
			suitability TEXT,
			explanation TEXT
		);
		"""
	
	cursor.execute(create_table)

	# execute the initial count query and retrieve the result
	initial_count_query = f"""
		SELECT COUNT(*) FROM {table_name}
	"""
	cursor.execute(initial_count_query)
	initial_count_result = cursor.fetchone()
	
	""" INDISCRIMANTELY INSERT THE VALUES """

	jobs_added = []
	for index, row in df.iterrows():
		insert_query = f"""
			INSERT INTO {table_name} (id, title, link, location, pubdate, summary, user_id, suitability, explanation)
			VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
			RETURNING *
		"""
		values = (row['id'], row['title'], row['link'], row['location'], row['pubdate'], row['summary'], row['user_id'], row['suitability'], row['explanation'])
		cursor.execute(insert_query, values)
		affected_rows = cursor.rowcount
		if affected_rows > 0:
			jobs_added.append(cursor.fetchone())


	""" LOGGING/PRINTING RESULTS"""

	final_count_query = f"""
		SELECT COUNT(*) FROM {table_name}
	"""
	# execute the count query and retrieve the result
	cursor.execute(final_count_query)
	final_count_result = cursor.fetchone()

	# calculate the number of unique jobs that were added
	if initial_count_result is not None:
		initial_count = initial_count_result[0]
	else:
		initial_count = 0
	jobs_added_count = len(jobs_added)
	if final_count_result is not None:
		final_count = final_count_result[0]
	else:
		final_count = 0

	elapsed_time = timeit.default_timer() - start_time

	postgre_report = f"""
		{table_name} report:\n
		Total count of jobs before crawling: {initial_count}
		Total number of unique jobs: {jobs_added_count}
		Current total count of jobs in PostgreSQL: {final_count}
		Duration: {elapsed_time:.2f}
		"""
	logging.info(postgre_report)



delimiters = "####"

system_prompt=f"""

You are a job recruiter for a large recruitment agency./
You will be provided with a candidate's CV./
The CV will be delimited with {delimiters} characters./
You will also be provided with the Job IDs (delimited by angle brackets) /
and corresponding descriptions (delimited by triple dashes)/
for the available job openings./

Perform the following steps:/

Step 1 - Classify the provided CV into a suitability category for each job opening./
Step 2 - For each ID briefly explain in one sentence your reasoning behind the chosen suitability category./
Step 3 - Only provide your output in json format with the keys: id, suitability and explanation./

Do not classify a CV into a suitability category until you have classify the CV yourself.

Suitability categories: Highly Suitable, Moderately Suitable, Potentially Suitable, Marginally Suitable and Not Suitable./

Highly Suitable: CVs in this category closely align with the job opening, demonstrating extensive relevant experience, skills, and qualifications. The candidate possesses all or most of the necessary requirements and is an excellent fit for the role./
Moderately Suitable: CVs falling into this category show a reasonable match to the job opening. The candidate possesses some relevant experience, skills, and qualifications that align with the role, but there may be minor gaps or areas for improvement. With some additional training or development, they could become an effective candidate./
Potentially Suitable: CVs in this category exhibit potential and may possess transferable skills or experience that could be valuable for the job opening. Although they may not meet all the specific requirements, their overall profile suggests that they could excel with the right support and training./
Marginally Suitable: CVs falling into this category show limited alignment with the job opening. The candidate possesses a few relevant skills or experience, but there are significant gaps or deficiencies in their qualifications. They may require substantial training or experience to meet the requirements of the role./
Not Suitable: CVs in this category do not match the requirements and qualifications of the job opening. The candidate lacks the necessary skills, experience, or qualifications, making them unsuitable for the role./
"""

introduction_prompt = """


\n Available job openings:\n

"""

cv = """ Qualifications:
- LLB Law from the University of Bristol (2022 - present)
- Member of the Honours Program at UDLAP, researching FinTech, Financial Inclusion, Blockchain, Distributed Ledger Technologies, Cryptocurrencies, and Smart Contracts
- TOEFL® iBT score of 107 out of 120

Previous job titles:
- Data Analyst at Tata Consultancy Services México (June 2022 – September 2022)
- Legal Assistant at BLACKSHIIP Venture Capital (May 2022 – July 2022)
- Data Analyst Jr. at AMATL GRÁFICOS (January 2020 – May 2022)
- Mathematics Instructor at ALOHA Mental Arithmetic (December 2019 – January 2020)
- Special Needs Counsellor at Camp Merrywood (O)
- Special Needs Counsellor at Camp Merrywood (Ontario, Canada) (May 2019 - August 2019)
- Special Needs Counsellor at YMCA Camp Independence (Chicago, USA) (June 2018 - August 2018)
- Coordinator of Volunteers at NAHUI OLLIN (November 2017 - May 2019)

Responsibilities/Key Duties:
- Cleansed, interpreted, and analyzed data with Python and SQL Server to produce visual reports using Power BI
- Proofread, drafted, and simplified legal documents such as Memorandums of Understanding, Terms & Conditions, Data Processing Agreements, Privacy Policies, etc.
- Developed and introduced A/B testing to make data-backed decisions and achieve increased Net Profit Margin
- Taught mental arithmetic to students and trained gifted children for national competitions
- Led and assisted individuals with physical and mental disabilities in camp settings
- Coordinated and supervised volunteers for an organization, increasing the number of volunteers by 400%

Skills:
- Written and verbal communication skills
- Teamwork and ability to work under pressure
- Attention to detail and judgment
- Leadership and people skills
- Python, SQL Server, MySQL/PostgreSQL, Tableau, Power BI, Bash/Command Line, Git & GitHub, Office 365, Machine Learning, Probabilities & Statistics

Other Achievements:
- Published paper on Smart Legal Contracts: From Theory to Reality
- Participated in the IDEAS Summer Program on Intelligence, Data, Ethics, and Society at the University of California, San Diego. 

"""