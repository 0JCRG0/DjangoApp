import os
import openai
import psycopg2
import pandas as pd
from scipy import spatial
import pretty_errors
import timeit
import logging
import time
import asyncio
from typing import Dict, Any
import json
from dotenv import load_dotenv
import logging
from aiohttp import ClientSession
from typing import Tuple
import re
import tiktoken
import pandas as pd
from datetime import datetime, timedelta
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pyarrow.parquet as pq
from aiohttp import ClientSession
import os

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
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


#Start the timer
start_time = timeit.default_timer()

""""
Load the embedded file
"""

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

def count_words(text: str) -> int:
	# Remove leading and trailing whitespaces
	text = text.strip()

	# Split the text into words using whitespace as a delimiter
	words = text.split()

	# Return the count of words
	return len(words)

async def async_summarise_job_gpt(session, job_description: str) -> Tuple[str, float]:
	await asyncio.sleep(.5)
	openai.aiosession.set(session)
	response = await openai.ChatCompletion.acreate(
		messages=[
			{'role': 'user', 'content': system_prompt_summary},
			{'role': 'user', 'content': f"Job Opening: {delimiters_summary}{job_description}{delimiters_summary}"},
		],
		model=MODEL,
		temperature=0,
		max_tokens = 400
	)
	response_message = response['choices'][0]['message']['content']
	total_cost = 0
	prompt_tokens = response['usage']['prompt_tokens']
	completion_tokens = response['usage']['completion_tokens']
	#print(f"\nSUMMARISE JOBS FUNCTION\n", f"\nPROMPT TOKENS USED:{prompt_tokens}\n", f"COMPLETION TOKENS USED:{completion_tokens}\n" )
	#Approximate cost
	if MODEL == "gpt-3.5-turbo":
		prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
		completion_cost = round((completion_tokens / 1000) * 0.002, 3)
		total_cost = prompt_cost + completion_cost
		#print(f"COST FOR SUMMARISING: ${total_cost:.2f} USD")
	elif MODEL == "gpt-3.5-turbo-16k":
		prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
		completion_cost = round((completion_tokens / 1000) * 0.004, 3)
		total_cost = prompt_cost + completion_cost
		#print(f"COST FOR SUMMARISING: ${total_cost:.2f} USD")
	return response_message, total_cost

async def async_summarise_description(description: str) -> tuple:
	#start timer
	start_time = asyncio.get_event_loop().time()
	total_cost = 0

	async def process_description(session, text):
		attempts = 0
		while attempts < 5:
			try:
				words_per_text = count_words(text)
				if words_per_text > 50:
					description_summary, cost = await async_summarise_job_gpt(session, text)
					return description_summary, cost
				else:
					logging.warning(f"Description is too short for being summarised. Number of words: {words_per_text}")
					return text, 0
			except (Exception, ServiceUnavailableError) as e:
				attempts += 1
				print(f"{e}. Retrying attempt {attempts}...")
				logging.warning(f"{e}. Retrying attempt {attempts}...")
				await asyncio.sleep(5**attempts)  # exponential backoff
		else:
			print(f"Description could not be summarised after 5 attempts.")
			return text, 0

	async with ClientSession() as session:
		result = await process_description(session, description)

	total_cost = result[1]

	#await close_session()
	#processed_time = timeit.default_timer() - start_time
	elapsed_time = asyncio.get_event_loop().time() - start_time

	return result[0], total_cost, elapsed_time

def num_tokens(text: str, model: str ="gpt-3.5-turbo") -> int:
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

def e5_base_v2_query(query):
	tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
	model = AutoModel.from_pretrained('intfloat/e5-base-v2')

	query_e5_format = f"query: {query}"

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
		logging.info("json_output_GPT4 is a list of dictionaries. Continuing...")
		df_json_output_GPT4 = pd.read_json(json.dumps(json_output_GPT4))
		return df_json_output_GPT4	
	elif isinstance(json_output_GPT4, dict):
		# If json_output_GPT4 is a dictionary, it contains a single record
		logging.info("json_output_GPT4 is a dictionary.\nEither contains a single record or gpt fucked up")
		data = [json_output_GPT4]
		df = pd.DataFrame(data)
		#df['id'] = df['id'].astype(int)
		return df
	else:
		# Handle other cases if necessary
		pass

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


"""   











START













"""


LoggingDjango()

async def main(user_id:int, user_country_1:str, user_country_2:str, user_cv:str, top_n_interval:int, num_suitable_jobs: int):

	if user_cv:
		user_cv_bool = True
	else:
		user_cv_bool = False

	logging.info(f"USER ID: {user_id}. USER DESIRED COUNTRY: {user_country_1} & {user_country_2}. USER CV: {user_cv_bool}")

	df_unfiltered = pd.read_parquet(E5_BASE_V2_DATA)

	df_two_weeks = filter_last_two_weeks(df_unfiltered)

	df = filter_df_per_countries(df=df_two_weeks, user_desired_country=user_country_1, user_second_desired_country=user_country_2)
	
	logging.info(f"Output of filter_df_per_countries(): {df} \n")

	def ids_ranked_by_relatedness_e5(query: str,
		df: pd.DataFrame,
		min_n: int,
		top_n: int,
		relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
	) -> tuple[list[str], list[float]]:
		
		#the query is embedded using e5
		query_embedding = e5_base_v2_query(query=query)

		ids_and_relatednesses = [
			(row["id"], relatedness_fn(query_embedding, row["embedding"]))
			for i, row in df.iterrows()
		]
		ids_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
		ids, relatednesses = zip(*ids_and_relatednesses)
		return ids[min_n:top_n], relatednesses[min_n:top_n]     
		#Returns a list of strings and relatednesses, sorted from most related to least.

	async def async_query_summary(
		query: str,
		df: pd.DataFrame,
		model: str,
		token_budget: int,
		min_n: int,
		top_n: int
	) -> str:
		#Return a message for GPT, with relevant source texts pulled from a dataframe.
		ids, relatednesses = ids_ranked_by_relatedness_e5(query, df, min_n=min_n, top_n=top_n)
		if ids:
			#Basically giving the most relevant IDs from the previous function
			introduction = introduction_prompt
			query_user = f"{query}"
			message = introduction
			# Create a list of tasks
			tasks = [async_summarise_description(df[df['id'] == id]['description'].values[0]) for id in ids]

			# Run the tasks concurrently
			results = await asyncio.gather(*tasks)
			job_summaries = []
			total_cost_summaries = 0    

			for id, result in zip(ids, results):
				job_description_summary, cost, elapsed_time = result
				
				# Append summary to the list
				job_summaries.append({
					"id": id,
					"summary": job_description_summary,
					"user_id": user_id
				})

				#all_users_ids.extend([user_id] * len(job_summaries))

				#Append total cost
				total_cost_summaries += cost

				next_id = f'\nID:<{id}>\nJob Description:---{job_description_summary}---\n'
				if (
					num_tokens(message + next_id + query_user, model=model)
					> token_budget
				):
					break
				else:
					message += next_id
			return query_user, message, job_summaries, total_cost_summaries

	#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
	async def ask(
		#This query is your question, only parameter to fill in function
		query: str,
		min_n: int,
		top_n: int,
		df: pd.DataFrame = df,
		model: str = GPT_MODEL,
		token_budget: int = 8192,
		log_gpt_messages: bool = True,
	) -> str:
		#Answers a query using GPT and a dataframe of relevant texts and embeddings.
		query_user, job_id_description, job_summaries, total_cost_summaries = await async_query_summary(query, df, model=model, token_budget=token_budget, min_n=min_n, top_n=top_n)

		#Save summaries in a df & then parquet -> append data if function called more than once
		df_summaries = pd.DataFrame(job_summaries)
		append_parquet(df_summaries, 'summaries')
		
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": f"{delimiters}{query_user}{delimiters}"},
			{"role": "assistant", "content": job_id_description}
		]
		
		if log_gpt_messages:
			logging.info(messages)
		response = openai.ChatCompletion.create(
			model=model,
			messages=messages,
			temperature=0
		)
		response_message = response["choices"][0]["message"]["content"]
		
		#if print_cost_and_relatednesses:
		total_tokens = response['usage']['total_tokens']
		prompt_tokens = response['usage']['prompt_tokens']
		completion_tokens = response['usage']['completion_tokens']
		
		logging.info(f"\nOPERATION: GPT-3.5 TURBO SUMMARISING. \nTOTAL COST: ${total_cost_summaries} USD")

		#Approximate cost
		if GPT_MODEL == "gpt-4":
			prompt_cost = round((prompt_tokens / 1000) * 0.03, 3)
			completion_cost = round((completion_tokens / 1000) * 0.06, 3)
			cost_classify = prompt_cost + completion_cost
			logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")
		elif GPT_MODEL == "gpt-3.5-turbo":
			prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
			completion_cost = round((completion_tokens / 1000) * 0.002, 3)
			cost_classify = prompt_cost + completion_cost
			logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")
		elif GPT_MODEL == "gpt-3.5-turbo-16k":
			prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
			completion_cost = round((completion_tokens / 1000) * 0.004, 3)
			cost_classify = prompt_cost + completion_cost
			logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")

		#relatednesses
		ids, relatednesses = ids_ranked_by_relatedness_e5(query=query, df=df, min_n=min_n, top_n=top_n)
		for id, relatedness in zip(ids, relatednesses):
			logging.info(f"ID: {id} has the following {relatedness=:.3f}")
		
		elapsed_time = (timeit.default_timer() - start_time) / 60
		logging.info(f"\nGPT-3.5 TURBO & GPT-4 finished summarising and classifying! all in: {elapsed_time:.2f} minutes \n")
		
		return response_message

	async def check_output_GPT4(input_cv: str, min_n:int, top_n:int) -> str:
		default = '[{"id": "", "suitability": "", "explanation": ""}]'
		default_json = json.loads(default)
		
		for _ in range(6):
			i = _ + 1
			try:
				python_string = await ask(query=input_cv, min_n=min_n, top_n=top_n)
				try:
					data = json.loads(python_string)
					logging.info(f"Response is a valid json object. Done in loop number: {i}")
					return data
				except json.JSONDecodeError:
					pass
			except openai.RateLimitError as e:
				logging.warning(f"{e}. Retrying in 10 seconds. Number of retries: {i}")
				time.sleep(10)
				pass
			except Exception as e:
				logging.warning(f"{e}. Retrying in 5 seconds. Number of retries: {i}", exc_info=True)
				time.sleep(5)
				pass

		logging.error("Check logs!!!! Main function was not callable. Setting json to default")
		return default_json

	#Modify df options - useful for logging
	set_dataframe_display_options()

	#Define the rows to classify
	min_n=0
	top_n=top_n_interval

	# Define the suitable categories
	suitable_categories = ['Highly Suitable', 'Moderately Suitable', 'Potentially Suitable']

	# Initialize the dataframe
	df_appended = pd.DataFrame()

	# Continue to call the function until we have 10 suitable jobs
	counter = 0
	while True:
		json_output_GPT4 = await check_output_GPT4(input_cv=user_cv, min_n=min_n, top_n=top_n)
		
		# Convert the JSON to a dataframe and append it to the existing dataframe
		df_json_output_GPT4 = pd.read_json(json.dumps(json_output_GPT4))
		df_appended = pd.concat([df_appended, df_json_output_GPT4], ignore_index=True)
		
		counter += 1
		logging.info(f"Looking for suitable jobs. Current loop: {counter}")

		logging.info(f"Current min_n: {min_n}. Current top_n: {top_n}")

		# Increment the counters depending on the desired top_n_interval 
		min_n += top_n_interval
		top_n += top_n_interval

		# Filter the dataframe to only include the suitable jobs
		df_most_suitable = df_appended[df_appended['suitability'].isin(suitable_categories)] if 'suitability' in df_appended.columns else pd.DataFrame()
		
		df_appended.to_parquet(SAVE_PATH + "/df_appended.parquet", index=False)
		df_most_suitable.to_parquet(SAVE_PATH + "/df_most_suitable.parquet", index=False)

		# Break the loop if we have x suitable jobs
		if len(df_most_suitable) >= num_suitable_jobs:
			break

	logging.info(f"\nDF APPENDED:\n{df_appended}. \nDF MOST SUITABLE:\n{df_most_suitable}")
	
	#Get the ids
	def ids_df_most_suitable(df: pd.DataFrame = df_most_suitable) -> str:
		ids = ""
		for _, row in df.iterrows():
			if "id" in row:
				if ids:
					ids += ", "
				ids += f"'{row['id']}'"

		return f"({ids})"

	ids_most_suitable = ids_df_most_suitable()
	logging.info(f"Getting the ids from the json object: {type(ids_most_suitable)}, {ids_most_suitable}")

	def find_jobs_per_ids(ids:str, table: str = "main_jobs") -> pd.DataFrame:
		conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
		# Create a cursor object
		cur = conn.cursor()
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
				# Close the database connection
		cur.close()
		conn.close()

		return df

	df_postgre = find_jobs_per_ids(ids=ids_most_suitable)

	#Read the parquet with ids, summaries & user_id
	df_summaries = pd.read_parquet(SAVE_PATH + "/summaries.parquet")
	#Merge it with the data in postgre
	df_postgre_summaries = df_postgre.merge(df_summaries, on='id', how='inner')
	#Merge with most suitable df so you have all the rows
	df = df_postgre_summaries.merge(df_most_suitable, on="id", how='inner')

	logging.info(f"\nALL ROWS:\n{df}")


	def sort_df_by_suitability(df: pd.DataFrame = df) -> pd.DataFrame:
		custom_order = {
			'Highly Suitable': 1,
			'Moderately Suitable': 2,
			'Potentially Suitable': 3
		}
		df['suitability_rank'] = df['suitability'].map(custom_order)
		sorted_df = df.sort_values(by='suitability_rank')
		sorted_df = sorted_df.drop(columns='suitability_rank')
		return sorted_df

	sorted_df = sort_df_by_suitability()

	filename = "/final_user_df"
	
	sorted_df.to_parquet(SAVE_PATH + f"{filename}.parquet", index=False)

	logging.info(f"\nSORTED DF:\n{sorted_df}.\n\nThis df has been saved in ...{filename}.parquet\n\n\n")
	
	return sorted_df



"""  








ADDITIONAL SUITABLE JOBS









"""



async def additional_suitable_jobs(user_id:int, user_country_1:str, user_country_2:str, user_cv:str, top_n_interval:int, num_suitable_jobs: int):

	if user_cv:
		user_cv_bool = True
	else:
		user_cv_bool = False

	logging.info(f"USER ID: {user_id}. USER DESIRED COUNTRY: {user_country_1} & {user_country_2}. USER CV: {user_cv_bool}")

	df_unfiltered = pd.read_parquet(E5_BASE_V2_DATA)

	df_two_weeks = filter_last_two_weeks(df_unfiltered)
	df_two_weeks_total_entries = df_two_weeks.count().sum()
	if df_two_weeks_total_entries > 0:
		df = filter_df_per_countries(df=df_two_weeks, user_desired_country=user_country_1, user_second_desired_country=user_country_2)
	else:
		logging.error("""THERE ARE NO ENTRIES IN "df_two_weeks". \nPopulate E5_BASE_V2_DATA with current jobs ASAP. """)
		raise

	def ids_ranked_by_relatedness_e5(query: str,
		df: pd.DataFrame,
		min_n: int,
		top_n: int,
		relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
	) -> tuple[list[str], list[float]]:
		
		#the query is embedded using e5
		query_embedding = e5_base_v2_query(query=query)

		ids_and_relatednesses = [
			(row["id"], relatedness_fn(query_embedding, row["embedding"]))
			for i, row in df.iterrows()
		]
		ids_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
		ids, relatednesses = zip(*ids_and_relatednesses)
		return ids[min_n:top_n], relatednesses[min_n:top_n]     
		#Returns a list of strings and relatednesses, sorted from most related to least.

	async def async_query_summary(
		query: str,
		df: pd.DataFrame,
		model: str,
		token_budget: int,
		min_n: int,
		top_n: int
	) -> str:
		#Return a message for GPT, with relevant source texts pulled from a dataframe.
		top_n_ids, relatednesses = ids_ranked_by_relatedness_e5(query, df, min_n=min_n, top_n=top_n)
		logging.info(f"IDs FROM ids_ranked_by_relatedness_e5(): {top_n_ids} \nNumber of IDs: {len(top_n_ids)}")
		
		#Get the preexisting ids from postgre to check against the ids from ids_ranked_by_relatedness_e5()
		preexisting_ids = preexisting_ids_postgre(user_id=user_id)
		logging.info(f"IDs FROM preexisting_ids_postgre(): {preexisting_ids} \nNumber of IDs: {len(preexisting_ids)}")

		#Get unique ids
		ids = find_unique_ids(ids=top_n_ids, preexisting_ids=preexisting_ids)
		logging.info(f"Unique IDs: {ids} \nNumber of IDs: {len(ids)}")

		#Basically giving the most relevant IDs from the previous function
		introduction = introduction_prompt
		query_user = f"{query}"
		message = introduction
		# Create a list of tasks
		tasks = [async_summarise_description(df[df['id'] == id]['description'].values[0]) for id in ids]

		# Run the tasks concurrently
		results = await asyncio.gather(*tasks)
		job_summaries = []
		total_cost_summaries = 0    

		for id, result in zip(ids, results):
			job_description_summary, cost, elapsed_time = result
			
			# Append summary to the list
			job_summaries.append({
				"id": id,
				"summary": job_description_summary,
				"user_id": user_id
			})

			#all_users_ids.extend([user_id] * len(job_summaries))

			#Append total cost
			total_cost_summaries += cost

			next_id = f'\nID:<{id}>\nJob Description:---{job_description_summary}---\n'
			if (
				num_tokens(message + next_id + query_user, model=model)
				> token_budget
			):
				break
			else:
				message += next_id
		return query_user, message, job_summaries, total_cost_summaries

	#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
	async def ask(
		#This query is your question, only parameter to fill in function
		query: str,
		min_n: int,
		top_n: int,
		df: pd.DataFrame = df,
		model: str = GPT_MODEL,
		token_budget: int = 8192,
		log_gpt_messages: bool = True,
	) -> str:
		#Answers a query using GPT and a dataframe of relevant texts and embeddings.
		query_user, job_id_description, job_summaries, total_cost_summaries = await async_query_summary(query, df, model=model, token_budget=token_budget, min_n=min_n, top_n=top_n)

		#Save summaries in a df & then parquet -> append data if function called more than once
		df_summaries = pd.DataFrame(job_summaries)
		append_parquet(df_summaries, 'summaries')
		
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": f"{delimiters}{query_user}{delimiters}"},
			{"role": "assistant", "content": job_id_description}
		]
		
		if log_gpt_messages:
			logging.info(messages)
		response = openai.ChatCompletion.create(
			model=model,
			messages=messages,
			temperature=0
		)
		response_message = response["choices"][0]["message"]["content"]
		
		#if print_cost_and_relatednesses:
		total_tokens = response['usage']['total_tokens']
		prompt_tokens = response['usage']['prompt_tokens']
		completion_tokens = response['usage']['completion_tokens']
		logging.info(f"\nOPERATION: GPT-3.5 TURBO SUMMARISING. \nTOTAL COST: ${total_cost_summaries} USD")

		#Approximate cost
		if GPT_MODEL == "gpt-4":
			prompt_cost = round((prompt_tokens / 1000) * 0.03, 3)
			completion_cost = round((completion_tokens / 1000) * 0.06, 3)
			cost_classify = prompt_cost + completion_cost
			logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")
		elif GPT_MODEL == "gpt-3.5-turbo":
			prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
			completion_cost = round((completion_tokens / 1000) * 0.002, 3)
			cost_classify = prompt_cost + completion_cost
			logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")
		elif GPT_MODEL == "gpt-3.5-turbo-16k":
			prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
			completion_cost = round((completion_tokens / 1000) * 0.004, 3)
			cost_classify = prompt_cost + completion_cost
			logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")

		#relatednesses
		ids, relatednesses = ids_ranked_by_relatedness_e5(query=query, df=df, min_n=min_n, top_n=top_n)
		for id, relatedness in zip(ids, relatednesses):
			logging.info(f"ID: {id} has the following {relatedness=:.3f}")
		
		elapsed_time = (timeit.default_timer() - start_time) / 60
		logging.info(f"\nGPT-3.5 TURBO & GPT-4 finished summarising and classifying! all in: {elapsed_time:.2f} minutes \n")
		
		return response_message

	async def ensure_json_object(input_cv: str, min_n: int, top_n: int) -> dict:
		default = '[{"id": "", "suitability": "", "explanation": ""}]'

		for i in range(1, 5):
			try:
				python_string = await ask(query=input_cv, min_n=min_n, top_n=top_n)
				data = json.loads(python_string)
				logging.info(f"Response is a valid JSON object. Done in loop number: {i}")
				return data
			except openai.RateLimitError as e:
				logging.warning(f"{e}. Retrying in 10 seconds. Number of retries: {i}")
				time.sleep(10)
			except Exception as e:
				logging.warning(f"{e}. Retrying in 5 seconds. Number of retries: {i}", exc_info=True)
				time.sleep(5)

		logging.error("Setting JSON to default. Look at async_classify_jobs_gpt_4().")
		return json.loads(default)

	ensure_json_object()
	#Modify df options - useful for logging
	set_dataframe_display_options()

	#Define the rows to classify
	min_n=0
	top_n=top_n_interval

	# Define the suitable categories
	suitable_categories = ['Highly Suitable', 'Moderately Suitable', 'Potentially Suitable']

	# Initialize the dataframe
	df_appended = pd.DataFrame()

	# Continue to call the function until we have 10 suitable jobs
	counter = 0
	while True:
		json_output_GPT4 = await check_output_GPT4(input_cv=user_cv, min_n=min_n, top_n=top_n)
		
		#Check whether json_output_GPT4 is a list or dict
		df_json_output_GPT4 = list_or_dict_json_output_GPT4(json_output_GPT4)
		# Convert the JSON to a dataframe and append it to the existing dataframe
		#df_json_output_GPT4 = pd.read_json(json.dumps(json_output_GPT4))
		df_appended = pd.concat([df_appended, df_json_output_GPT4], ignore_index=True)
		df_appended['id'] = df_appended['id'].astype(int)

		counter += 1
		logging.info(f"Looking for suitable jobs. Current loop: {counter}")

		logging.info(f"Current min_n: {min_n}. Current top_n: {top_n}")

		# Increment the counters depending on the desired top_n_interval 
		min_n += top_n_interval
		top_n += top_n_interval

		# Filter the dataframe to only include the suitable jobs
		df_most_suitable = df_appended[df_appended['suitability'].isin(suitable_categories)] if 'suitability' in df_appended.columns else pd.DataFrame()
		
		df_appended.to_parquet(SAVE_PATH + "/df_appended.parquet", index=False)
		df_most_suitable.to_parquet(SAVE_PATH + "/df_most_suitable.parquet", index=False)

		# Break the loop if we have x suitable jobs
		if len(df_most_suitable) >= num_suitable_jobs:
			break

	logging.info(f"\nDF APPENDED:\n{df_appended}. \nDF MOST SUITABLE:\n{df_most_suitable}")
	
	#Get the ids
	def ids_df_most_suitable(df: pd.DataFrame = df_most_suitable) -> str:
		ids = ""
		for _, row in df.iterrows():
			if "id" in row:
				if ids:
					ids += ", "
				ids += f"'{row['id']}'"

		return f"({ids})"

	ids_most_suitable = ids_df_most_suitable()
	logging.info(f"Getting the ids from the json object: {type(ids_most_suitable)}, {ids_most_suitable}")

	def find_jobs_per_ids(ids:str, table: str = "main_jobs") -> pd.DataFrame:
		conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
		# Create a cursor object
		cur = conn.cursor()
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
				# Close the database connection
		cur.close()
		conn.close()

		return df

	df_postgre = find_jobs_per_ids(ids=ids_most_suitable)

	#Read the parquet with ids, summaries & user_id
	df_summaries = pd.read_parquet(SAVE_PATH + "/summaries.parquet")
	#Merge it with the data in postgre
	df_postgre_summaries = df_postgre.merge(df_summaries, on='id', how='inner')
	#Merge with most suitable df so you have all the rows
	df = df_postgre_summaries.merge(df_most_suitable, on="id", how='inner')

	logging.info(f"\nALL ROWS:\n{df}")


	def sort_df_by_suitability(df: pd.DataFrame = df) -> pd.DataFrame:
		custom_order = {
			'Highly Suitable': 1,
			'Moderately Suitable': 2,
			'Potentially Suitable': 3
		}
		df['suitability_rank'] = df['suitability'].map(custom_order)
		sorted_df = df.sort_values(by='suitability_rank')
		sorted_df = sorted_df.drop(columns='suitability_rank')
		return sorted_df

	sorted_df = sort_df_by_suitability()

	filename = "/final_user_df"
	
	sorted_df.to_parquet(SAVE_PATH + f"{filename}.parquet", index=False)

	logging.info(f"\nSORTED DF:\n{sorted_df}.\n\nThis df has been saved in ...{filename}.parquet\n\n\n")
	
	return sorted_df

if __name__ == "__main__":
	asyncio.run(main(user_id=40, user_country_1="Mexico", user_country_2="Anywhere", user_cv=True, top_n_interval=4, num_suitable_jobs=1))
	#asyncio.run(additional_suitable_jobs(user_id=37, user_country="Mexico", user_cv=cv, top_n_interval=4, num_suitable_jobs=1))
