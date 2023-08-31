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
from openai.error import OpenAIError
import json
from dotenv import load_dotenv
from openai.error import ServiceUnavailableError
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
