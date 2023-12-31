import os
import openai
import psycopg2
from psycopg2.extensions import cursor
import numpy as np
import pandas as pd
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
from aiohttp import ClientSession
from functools import wraps
from .prompts import *
import os

load_dotenv('.env')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")
SAVE_PATH = os.getenv("SAVE_PATH")
COUNTRIES_JSON_DATA = os.getenv("COUNTRIES_JSON_DATA")
LOGGER_DJANGO = os.getenv("LOGGER_DJANGO")


#Specs from 8/11/23
specs_gpt4_models = {
	"gpt-4-0613": (8192, 0.03, 0.06, "text"),
	"gpt-4-1106-preview": (128000, 0.01, 0.03, "json_object"),
	"gpt-4-vision-preview": (128000, 0.01, 0.03, "json_object"),
	"gpt-4": (8192, 0.03, 0.06, "text"),
	"gpt-4-32k": (32768, 0.06, 0.12, "text"),
	"gpt-4-32k-0613	": (32768, 0.06, 0.12, "text")
}

#Specs from 8/11/23
specs_gpt3_models = {
	"gpt-3.5-turbo-1106": (16385, 0.001, 0.002, "json_object"),
	"gpt-3.5-turbo": (4096, 0.0015, 0.002, "text"),
	"gpt-3.5-turbo-16k": (16385, 0.003, 0.004, "text")
}

#Specs from 8/11/23
specs_all_models = {
	"gpt-4-1106-preview": (128000, 0.01, 0.03, "json_object"),
	"gpt-4-vision-preview": (128000, 0.01, 0.03, "json_object"),
	"gpt-4": (8192, 0.03, 0.06, "text"),
	"gpt-4-0613": (8192, 0.03, 0.06, "text"),
	"gpt-4-32k": (32768, 0.06, 0.12, "text"),
	"gpt-4-32k-0613	": (32768, 0.06, 0.12, "text"),
	"gpt-3.5-turbo-1106": (16385, 0.001, 0.002, "json_object"),
	"gpt-3.5-turbo": (4096, 0.0015, 0.002, "text"),
	"gpt-3.5-turbo-16k": (16385, 0.003, 0.004, "text")
}

#----------------UTILS' UTILS----------------#

def count_words(text: str) -> int:
	# Remove leading and trailing whitespaces
	text = text.strip()

	# Split the text into words using whitespace as a delimiter
	words = text.split()

	# Return the count of words
	return len(words)

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


def whether_json_object(gpt4_response: str) -> bool:
	try:
		json.loads(gpt4_response)
		logging.info(f"Response is a valid JSON object. Continuing...")
		return True
	except json.JSONDecodeError as e:
		logging.warning(f"JSON decoding error: {e}.\nRetrying async_classify_jobs_gpt_4()\n\n", exc_info=True)
		return False

def list_or_dict_python_object(gpt4_response_python_object: object) -> pd.DataFrame:
	if isinstance(gpt4_response_python_object, list):
		logging.info("gpt4_response_python_object is a list of dictionaries. Continuing...")
		df_gpt4_response = pd.DataFrame(gpt4_response_python_object)
		df_gpt4_response['id'] = df_gpt4_response['id'].astype(int)
		return df_gpt4_response	
	elif isinstance(gpt4_response_python_object, dict):
		logging.info("gpt4_response_python_object is a dictionary.\nEither contains a single record or gpt fucked up")
		data = [gpt4_response_python_object]
		df_gpt4_response = pd.DataFrame(data)
		df_gpt4_response['id'] = df_gpt4_response['id'].astype(int)
		return df_gpt4_response
	else:
		# Handle other cases if necessary
		pass
#----------------MAIN UTILS----------------#

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


def fetch_similar_jobs_not_matched(
		user_id: int,
		user_cv_embedding: np.ndarray,
		cursor: cursor,
		all_country_values:str,
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

	main_query = f"""
    SELECT *
    FROM {table_name}
    WHERE NOT EXISTS (
        SELECT id
        FROM matched_jobs
        WHERE id = {user_id}
    ) AND timestamp >= (current_date - interval {interval_days})
    AND substring(lower(job_info) from '#### location: (.*?) ####') = ANY(%s::text[])
    ORDER BY embedding {similarity_metric} %s;
	"""
	
	cursor.execute(main_query.format(country_values_str, user_cv_embedding))

	# Fetch all the rows
	rows = cursor.fetchall()

	# Separate the columns into individual lists
	ids = [row[0] for row in rows]
	jobs_info = [row[1] for row in rows]

	df = pd.DataFrame({'id': ids, 'job_info': jobs_info})

	return df



def fetch_similar_jobs(
		user_cv_embedding: np.ndarray,
		all_country_values:str,
		cursor: cursor,
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

def retry_on_error_summarise_wrapper(func):
	@wraps(func)
	async def wrapper(*args, **kwargs):
		for model in specs_gpt3_models.keys():
			try:
				kwargs['summarize_gpt_model'] = model  # Add the model as a keyword argument
				return await asyncio.wait_for(func(*args, **kwargs), timeout=15)
			except (asyncio.TimeoutError, Exception) as e:
				time.sleep(.5)
				logging.warning(f"Error while summarizing with model {model}:\n\n{e}\nSleeping & Retrying.")
				continue
		logging.error(f"All the following models encountered errors while summarising:\n\n{specs_gpt3_models}.")
		raise Exception(f"All the models encountered errors while summarizing.")
	return wrapper

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

	if gpt_model in specs_gpt3_models:
		prompt_cost_per_k = specs_gpt3_models[gpt_model][1]
		completion_cost_per_k = specs_gpt3_models[gpt_model][2]
		prompt_cost = round((prompt_tokens / 1000) * prompt_cost_per_k, 3)
		completion_cost = round((completion_tokens / 1000) * completion_cost_per_k, 3)
		cost_per_summary = prompt_cost + completion_cost
		# logging.info(f"COST FOR SUMMARISING: ${total_cost:.4f} USD")
	else:
		logging.error("The gpt_model selected in invalid. Choose a valid option. See https://openai.com/blog/new-models-and-developer-products-announced-at-devday")
		raise Exception("The gpt_model selected in invalid. Choose a valid option. See https://openai.com/blog/new-models-and-developer-products-announced-at-devday")
	return response_message, cost_per_summary

async def async_summarise_description(description: str, gpt_model: str) -> tuple:

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

@retry_on_error_summarise_wrapper
async def async_format_top_jobs_summarize(
	user_id: int,
	user_cv: str,
	df: pd.DataFrame,
	classify_gpt_model: str,
	summarize_gpt_model=None
) -> Tuple[str, list[str], int]:
	
	logging.info(f"""\nUSING: "{summarize_gpt_model}" FOR SUMMARISING""")

	if classify_gpt_model in specs_all_models:
		token_budget = specs_all_models[classify_gpt_model][0]
	
	ids = df['id'].tolist()
	if ids:
		#Basically giving the most relevant IDs from the previous function
		message = introduction_prompt

		start_time = asyncio.get_event_loop().time()

		tasks = [async_summarise_description(df[df['id'] == id]['job_info'].values[0], summarize_gpt_model) for id in ids]

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

			next_id = f'\n<ID:{id}>\n---Job Description: {job_description_summary}---\n'
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
		logging.error("No IDs in async_format_top_jobs_summarize().\n If this happened in the first call then check fetch_top_n_matching_jobs().\nElse, we simply ran out of jobs to classify")
		raise TypeError ("IDs is empty so it won't unpack returned objects")

def retry_on_error_json_classify_wrapper(func):
	@wraps(func)
	async def wrapper(*args, **kwargs):
		result = None
		for model in specs_gpt4_models.keys():
			try:
				kwargs['classify_gpt_model'] = model 
				logging.info(f"")
				result = await asyncio.wait_for(func(*args, **kwargs), timeout=70)
				
				json.loads(result)
				
				return result
			except (asyncio.TimeoutError, Exception, json.JSONDecodeError) as e:
				time.sleep(.5)
				logging.warning(f"Error while classifying with model {model}:\n\n{e}\n\n{model} response: {result}.\nSleeping & Retrying with different model.")
				continue
		logging.error(f"All the following models encountered errors while classifying:\n\n{specs_gpt4_models}.")
		raise Exception(f"All the models encountered errors while classifying.")
	return wrapper

def retry_on_error_classify_wrapper(func):
	@wraps(func)
	async def wrapper(*args, **kwargs):
		result = None
		for model in specs_gpt4_models.keys():
			try:
				kwargs['classify_gpt_model'] = model 
				return await asyncio.wait_for(func(*args, **kwargs), timeout=70)
			except (asyncio.TimeoutError, Exception) as e:
				time.sleep(.5)
				logging.warning(f"Error while classifying with model {model}:\n\n{e}\n\n{model} response: {result}.\nSleeping & Retrying with different model.")
				continue
		logging.error(f"All the following models encountered errors while classifying:\n\n{specs_gpt4_models}.")
		raise Exception(f"All the models encountered errors while classifying.")
	return wrapper


@retry_on_error_classify_wrapper
async def async_classify_jobs_gpt_4(
	#This query is your question, only parameter to fill in function
	user_cv: str,
	formatted_message: str,
	log_gpt_messages: bool = True,
	classify_gpt_model=None,
	specs_all_models: dict=specs_all_models,
):
	logging.info(f"""\nCALLING: "{classify_gpt_model}" FOR CLASSIFYING TASK""")

	client = AsyncOpenAI(
		# defaults to os.environ.get("OPENAI_API_KEY")
		api_key=OPENAI_API_KEY,
	)

	start_time = asyncio.get_event_loop().time()

	if classify_gpt_model in specs_all_models:
		input_cost = specs_all_models[classify_gpt_model][1]
		output_cost = specs_all_models[classify_gpt_model][2]
		response_format_enabled = specs_all_models[classify_gpt_model][3]
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
	logging.info(f"""\nDONE CLASSIFYING.\n\nMODEL USED: "{classify_gpt_model}"\nINPUT:\nTOKENS USED: {prompt_tokens}.\nPROMP COST: ${prompt_cost}USD\n\nOUTPUT:\nTOKENS USED:{completion_tokens}\nCOMPLETION COST: {completion_cost}.\n\nTOTAL TOKENS USED:{total_tokens}\nTOTAL COST FOR CLASSIFYING: ${total_classifying_cost:.3f} USD""")

	elapsed_time = asyncio.get_event_loop().time() - start_time
	logging.info(f"""\n"{classify_gpt_model}" finished classifying! all in: {elapsed_time:.2f} seconds \n""")
	
	return response_message

async def retrying_async_classify_jobs_gpt_4(
		async_classify_jobs_gpt_4: Callable,
		user_cv: str,
		formatted_message: str,
		log_gpt_messages: bool,
		):
	
	default = [{"id": "", "suitability": "", "explanation": ""}]
	df_default = list_or_dict_python_object(default)

	#specs_gpt4_models is initialised in the beginning
	for model_name, model_specs in specs_gpt4_models.items():
		retries = 6  
		for i in range(retries):
			logging.info(f"""\nCalling retrying_async_classify_jobs_gpt_4().\nUsing "{model_name}" for loop number: {i + 1}...""")
			try:
				gpt4_response = await async_classify_jobs_gpt_4(
											user_cv,
											formatted_message,
											model_name,
											log_gpt_messages
										)
				try:
					python_object = json.loads(gpt4_response)
					logging.info(f"""Response is a valid json object.\nResponse: {gpt4_response}.\nModel used: "{model_name}".\nDone in loop number: {i + 1}""")
					df_gpt4_response = list_or_dict_python_object(python_object)
					return df_gpt4_response
				except json.JSONDecodeError:
					pass
			except openai.RateLimitError as e:
				logging.warning(f"{e}. Retrying in 10 seconds. Model: {model_name}, Number of retries: {i + 1}")
				time.sleep(10)
			except Exception as e:
				logging.warning(f"{e}. Retrying in 5 seconds. Model: {model_name}, Number of retries: {i + 1}", exc_info=True)
				time.sleep(5)

	logging.error("Check logs!!!! Main function was not callable. Setting df to default")
	return df_default


async def parse_response_async_classify_jobs_gpt_4(user_cv: str, formatted_message:str, log_gpt_messages:bool=True, attempts: int = 2) -> pd.DataFrame:
	for _ in range(attempts):
		try:
			response = await async_classify_jobs_gpt_4(user_cv, formatted_message, log_gpt_messages)
			type_gpt_4_response = type(response)
			logging.info(f"type_gpt_4_response: {type_gpt_4_response}.\ngpt_4_response: {response}")

			if whether_json_object(response):
				response_python_object = json.loads(response)
			else:
				try:
					response_python_object = eval(response)
				except SyntaxError as e:
					logging.warning(f"Response is possibly a string of a dict.\nSyntaxError: {e}\nTrying to wrap it in a list...")
					response_python_object = json.loads(f"[{response}]")
			return list_or_dict_python_object(response_python_object)
		except SyntaxError as e:
			logging.warning(f"Response is not a valid python object.\n{e}\nRetrying -> async_classify_jobs_gpt_4()...")
		except Exception as e:
			logging.error(f"URGENT! All the models encountered errors while classifying after calling twice.\n{e}.\nSetting the df to default)...")
			break

	default = [{"id": "", "suitability": "", "explanation": ""}]
	return list_or_dict_python_object(default)

async def parse_response_async_format_top_jobs_summarize(user_id: int, user_cv: str, sliced_df: pd.DataFrame, classify_gpt_model:str, attempts: int = 2) -> pd.DataFrame:
	for _ in range(attempts):
		try:
			formatted_message, job_summaries = await async_format_top_jobs_summarize(user_id, user_cv,sliced_df, classify_gpt_model)
			return formatted_message, job_summaries
		except TypeError as e:
			logging.error("No IDs in async_format_top_jobs_summarize().\n If this happened in the first call then check fetch_top_n_matching_jobs().\nElse, we simply ran out of jobs to classify.\nEither way, we are breaking out of the while loop.")
			break
		except Exception as e:
			logging.error(f"All the models encountered errors while summarizing.\n{e}.\nSleeping & Retrying -> async_format_top_jobs_summarize()...")
			await asyncio.sleep(10)
	
	logging.error(f"All attempts to call async_format_top_jobs_summarize failed.")



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

	df['id'] = df['id'].astype(int)

	return df

def postgre_insert_matched_jobs(cursor: cursor, df: pd.DataFrame(), table_name: str = "matched_jobs"):

	start_time = timeit.default_timer()

	create_table = f"""
		CREATE TABLE IF NOT EXISTS {table_name} (
			id INTEGER,
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
