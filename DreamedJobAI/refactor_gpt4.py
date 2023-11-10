import os
import openai
import psycopg2
import logging
import asyncio
import json
from dotenv import load_dotenv
import pandas as pd
from pgvector.psycopg2 import register_vector
from utils.main_utils import *

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")
LOCAL_POSTGRE_URL = os.environ.get("LOCAL_POSTGRE_URL")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LoggingDjango()

async def main(user_id: int, user_country_1: str, user_country_2: str | None, user_cv:str, limit_interval: int, num_suitable_jobs: int):
	
	start_time = asyncio.get_event_loop().time()

	logging.info(f"\nStarting main().\n\nArguments.\nuser_id: {user_id}.\nuser_country_1: {user_country_1}. user_country_2: {user_country_2}\nnum_suitable_jobs: {num_suitable_jobs}")

	#Get all the country values that match user's input
	all_country_values = fetch_all_country_values(user_country_1, user_country_2)
	
	#Embed user's query
	user_query_embedding = e5_base_v2_query(user_cv)

	#Make connection and enable pgvector
	conn = psycopg2.connect(LOCAL_POSTGRE_URL)
	cursor = conn.cursor()
	cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
	register_vector(conn)

	"""
	This function performs three actions:

	1. Filters user's country
	2. Filters to only query rows < 2 weeks from current date
	3. Performs similarity search depending on metric

	Returns a df containing the matching ids and respective jobs_info
	"""
	
	df = fetch_top_n_matching_jobs(user_query_embedding, all_country_values, cursor, top_n="1", similarity_or_distance_metric="NN")
	
	#TODO: Have a function that will output the distance and the rows.
	# To compare outputs depending on metric.
	# This is only for analytics, not functionality.

	#Initialise the range
	start = 0
	limit = 0
	
	max_number_rows_df = len(df)

	if limit > max_number_rows_df:
		limit_interval = max_number_rows_df

	limit = limit_interval

	logging.info(f"""Incremental values:\n\nStarting on row number {start}.\nStopping on row number {limit}.\nTotal number of rows from the df: {max_number_rows_df}""")

	# Define the suitable categories
	MOST_SUITABLE_CATEGORIES = ['Highly Suitable', 'Moderately Suitable', 'Potentially Suitable']
	
	# Initialize the dataframe
	accumulator_df = pd.DataFrame()
	
	# Continue to call the function until we have x suitable jobs
	counter = 0
	while max_number_rows_df > limit or True:
		
		#Only get the top_n from the original df
		sliced_df = df.iloc[start:limit]

		"""
		This function performs the following actions:

		1. Concurrently summarizes job descriptions 
		2. Tracks the cost of each summary.
		3. Constructs a message for GPT, including ids, 
		job summaries and user CV text, while respecting a token budget.
		
		Returns a tuple containing the constructed message and a list of job summaries
		"""
		#TODO: Wrapper that will change the model if time is > 10s
		formatted_message, job_summaries = await async_format_top_jobs_summarize(
																user_id,
																user_cv,
																sliced_df,
																summarize_gpt_model="gpt-3.5-turbo-1106",
																classify_gpt_model="gpt-3.5-turbo-1106"
															)

		
		df_summaries = pd.DataFrame(job_summaries)
		#append_parquet(df_summaries, 'summaries')
		
		"""
		This function asynchronously classifies job-related
		information using the GPT-4 model.
		
		Returns the GPT-4 model's generated response for the given input.
		"""

		#TODO: Wrapper that will change the model if time is > 70s
		gpt4_response = await async_classify_jobs_gpt_4(
													user_cv,
													formatted_message,
													classify_gpt_model = "gpt-3.5-turbo-1106",
													log_gpt_messages= True
												)
		logging.info(gpt4_response)
		
		#Whether the output is a json object
		json_object = whether_json_object(gpt4_response)

		"""
		If the output is not a json object.

		Try with all the different models.
		"""

		if json_object:
			gpt4_response_json_object = json.loads(gpt4_response)
		else:
			gpt4_response_json_object = await retrying_async_classify_jobs_gpt_4(async_classify_jobs_gpt_4, user_cv, formatted_message, log_gpt_messages=True)
		
		logging.info(f"""Results of iteration number {counter}:\n\n{gpt4_response_json_object}""")		
		
		df_gpt4_response_json_object = pd.DataFrame(len(gpt4_response_json_object), type(gpt4_response_json_object), gpt4_response_json_object)
		accumulator_df = pd.concat([accumulator_df, df_gpt4_response_json_object], ignore_index=True)
		
		# Filter the dataframe to only include the suitable jobs
		df_most_suitable = df_gpt4_response_json_object[df_gpt4_response_json_object['suitability'].isin(MOST_SUITABLE_CATEGORIES)] if 'suitability' in df_gpt4_response_json_object.columns else pd.DataFrame()

		logging.info(f"""Number of suitable jobs found so far: {len(df_most_suitable)}\nNumber of jobs to find: {num_suitable_jobs}""")
		
		# Increasing values for next iteration
		counter += 1
		start += limit_interval
		limit += limit_interval
		
		# Break the loop if we have x suitable jobs
		if len(df_most_suitable) >= num_suitable_jobs:
			logging.info(f"While loop is done.\nFound jobs: {len(df_most_suitable)}\nTarget: {num_suitable_jobs}")
			break
		
		logging.info(f"Increasing values for next iteration.\n\nLoop number {counter}\nStarting on row {start}.\nStopping on row {limit}")
	
	ids_most_suitable = ids_df_most_suitable(df=df_most_suitable)

	logging.info(f"IDs from df_most_suitable: {ids_most_suitable}")

	df_main_jobs = find_jobs_main_jobs_per_ids(cur=cursor, ids=ids_most_suitable)
	#Merge the ids, summaries & user_id with data in main_jobs
	df_main_jobs_summaries = df_main_jobs.merge(df_summaries, on='id', how='inner')
	#Merge with most suitable df so you have all the rows
	df_matched_jobs = df_main_jobs_summaries.merge(df_most_suitable, on="id", how='inner')

	postgre_insert_matched_jobs(cursor, df_matched_jobs)

	# Close the database connection
	conn.commit()
	cursor.close()
	conn.close()

	elapsed_time = asyncio.get_event_loop().time() - start_time

	logging.info(f"""\nmain() finished! all in: {elapsed_time:.2f} seconds \n""")


if __name__ == "__main__":
	asyncio.run(main(user_id=40, user_country_1="United States", user_country_2="Anywhere", user_cv=cv, limit_interval=5, num_suitable_jobs=2))
	#asyncio.run(main(top_n_interval=4, num_suitable_jobs=1))
