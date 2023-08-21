import os
import openai
from dotenv import load_dotenv
import pandas as pd
import pretty_errors
from openai.error import ServiceUnavailableError
import logging
from aiohttp import ClientSession
import asyncio
from typing import Tuple
import re
import tiktoken
import pandas as pd
import logging
import json
from datetime import datetime, timedelta
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pyarrow.parquet as pq
from aiohttp import ClientSession
from dotenv import load_dotenv
import os

load_dotenv(".env")
SAVE_PATH = os.getenv("SAVE_PATH")
COUNTRIES_JSON_DATA = os.getenv("COUNTRIES_JSON_DATA")
LOGGER_DJANGO = os.getenv("LOGGER_DJANGO")
openai.api_key = os.getenv("OPENAI_API_KEY")

#MODEL= "gpt-3.5-turbo-16k"
MODEL= "gpt-3.5-turbo"


delimiters = "----"
delimiters_job_info = '####'

system_query = f""" 

Your task is to extract the specified information from a job opening/
posted by a company, with the aim of effectively matching /
potential candidates for the position./

The job opening below is delimited by {delimiters} characters./
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
            {'role': 'user', 'content': system_query},
            {'role': 'user', 'content': f"Job Opening: {delimiters}{job_description}{delimiters}"},
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

"""
async def summarise_descriptions(descriptions: list) -> list:
	#start timer
	start_time = asyncio.get_event_loop().time()
	total_cost = 0

	async def process_description(session, i, text):
		attempts = 0
		while attempts < 5:
			try:
				words_per_text = count_words(text)
				if words_per_text > 50:
					description_summary, cost = await async_summarise_job_gpt(session, text)
					print(f"Description with index {i} just added.")
					logging.info(f"Description's index {i} just added.")
					return i, description_summary, cost
				else:
					logging.warning(f"Description with index {i} is too short for being summarised. Number of words: {words_per_text}")
					print(f"Description with index {i} is too short for being summarised. Number of words: {words_per_text}")
					return i, text, 0
			except (Exception, ServiceUnavailableError) as e:
				attempts += 1
				print(f"{e}. Retrying attempt {attempts}...")
				logging.warning(f"{e}. Retrying attempt {attempts}...")
				await asyncio.sleep(5**attempts)  # exponential backoff
		else:
			print(f"Description with index {i} could not be summarised after 5 attempts.")
			return i, text, 0

	async with ClientSession() as session:
		tasks = [process_description(session, i, text) for i, text in enumerate(descriptions)]
		results = await asyncio.gather(*tasks)

	# Sort the results by the index and extract the summaries and costs
	results.sort()
	descriptions_summarised = [result[1] for result in results]
	costs = [result[2] for result in results]
	total_cost = sum(costs)

	#await close_session()
	#processed_time = timeit.default_timer() - start_time
	elapsed_time = asyncio.get_event_loop().time() - start_time

	return descriptions_summarised, total_cost, elapsed_time
"""

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


load_dotenv(".env")
SAVE_PATH = os.getenv("SAVE_PATH")
COUNTRIES_JSON_DATA = os.getenv("COUNTRIES_JSON_DATA")
LOGGER_DJANGO = os.getenv("LOGGER_DJANGO")

def clean_rows(s):
	if not isinstance(s, str):
		print(f"{s} is not a string! Returning unmodified")
		return s
	s = re.sub(r'\(', '', s)
	s = re.sub(r'\)', '', s)
	s = re.sub(r"'", '', s)
	s = re.sub(r",", '', s)
	return s

def truncated_string(
	string: str,
	model: str,
	max_tokens: int,
	print_warning: bool = False,
) -> str:
	"""Truncate a string to a maximum number of tokens."""
	encoding = tiktoken.encoding_for_model(model)
	encoded_string = encoding.encode(string)
	truncated_string = encoding.decode(encoded_string[:max_tokens])
	if print_warning and len(encoded_string) > max_tokens:
		print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
	return truncated_string

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

def count_words(text: str) -> int:
	# Remove leading and trailing whitespaces
	text = text.strip()

	# Split the text into words using whitespace as a delimiter
	words = text.split()

	# Return the count of words
	return len(words)

def append_parquet(new_df: pd.DataFrame, filename: str):
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

def passage_e5_format(raw_descriptions):
	formatted_batches = ["passage: {}".format(raw_description) for raw_description in raw_descriptions]
	return formatted_batches

def set_dataframe_display_options():
    # Call the function to set the desired display options
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)  # Disable column width restriction
    pd.set_option('display.expand_frame_repr', False)  # Disable wrapping to multiple lines
    pd.set_option('display.max_colwidth', None)  # Display full contents of each column

def filter_df_per_country(df: pd.DataFrame, user_desired_country:str) -> pd.DataFrame:
	# Load the JSON file into a Python dictionary
	with open(COUNTRIES_JSON_DATA, 'r') as f:
		data = json.load(f)

	# Function to get country information
	def get_country_info(user_desired_country):
		values = []
		for continent, details in data.items():
			for country in details['Countries']:
				if country['country_name'] == user_desired_country:
					values.append(country['country_name'])
					values.append(country['country_code'])
					values.append(country['capital_english'])
					for subdivision in country['subdivisions']:
						values.append(subdivision['subdivisions_code'])
						values.append(subdivision['subdivisions_name'])
		return values

	# Get information for a specific country
	country_values = get_country_info(user_desired_country)

	# Convert 'location' column to lowercase
	df['location'] = df['location'].str.lower()

	# Convert all country values to lowercase
	country_values = [value.lower() for value in country_values]

	# Create a mask with all False
	mask = pd.Series(False, index=df.index)

	# Update the mask if 'location' column contains any of the country values
	for value in country_values:
		mask |= df['location'].str.contains(value, na=False)

	# Filter DataFrame
	filtered_df = df[mask]

	return filtered_df

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


abstract_cv_past = """Data Analyst: Cleansed, analyzed, and visualized data using Python, SQL Server, and Power BI.
Legal Assistant: Drafted legal documents, collaborated on negotiation outlines, and handled trademark registrations.
Data Analyst Jr.: Implemented A/B testing, utilized data analysis tools, and developed real-time visualizations.
Special Needs Counselor: Led and assisted individuals with disabilities, provided personal care, and facilitated camp activities.
Total years of professional experience: 3 years."""

abstract_cv = """('Qualifications: \n- LLB Law degree from Universidad de las Américas Puebla (UDLAP) with an accumulated average of 9.4/10.\n- Currently on an international exchange at the University of Bristol for the final year of studying Law.\n- Member of the Honours Program at UDLAP, conducting research on FinTech, Financial Inclusion, Blockchain, Cryptocurrencies, and Smart Contracts.\n\nPrevious job titles:\n- Data Analyst at Tata Consultancy Services México, where I cleansed, interpreted, and analyzed data using Python and SQL Server to produce visual reports with Power BI.\n- Legal Assistant at BLACKSHIIP Venture Capital, responsible for proofreading and drafting legal documents, as well as assisting with negotiations of International Share Purchase Agreements.\n\nResponsibilities/Key Duties:\n- Developed and introduced A/B testing to make data-driven business decisions as a Data Analyst Jr. at AMATL GRÁFICOS.\n- Taught mental arithmetic as a Mathematics Instructor at ALOHA Mental Arithmetic.\n- Led and assisted individuals with physical and mental disabilities as a Special Needs Counsellor at Camp Merrywood and YMCA Camp Independence.\n\nSkills:\n- Proficient in Python, SQL Server, Tableau, Power BI, Bash/Command Line, Git & GitHub, and Office 365.\n- Strong written and verbal communication skills, teamwork, ability to work under pressure, attention to detail, and leadership skills.\n- Knowledge in machine learning, probabilities & statistics, and proofreading.\n\nOther Achievements:\n- Published paper on "Smart Legal Contracts: From Theory to Reality" and participated in the IDEAS Summer Program on Intelligence, Data, Ethics, and Society at the University of California, San Diego."""