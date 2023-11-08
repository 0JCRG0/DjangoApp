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
from sqlalchemy import create_engine, text
from aiohttp import ClientSession
from pgvector.psycopg2 import register_vector
from utils.main_utils import *
import os

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")
LOCAL_POSTGRE_URL = os.environ.get("LOCAL_POSTGRE_URL")
SAVE_PATH = os.getenv("SAVE_PATH")
E5_BASE_V2_DATA = os.getenv("E5_BASE_V2_DATA")
COUNTRIES_JSON_DATA = os.getenv("COUNTRIES_JSON_DATA")
LOGGER_DJANGO = os.getenv("LOGGER_DJANGO")
LOGGER_DIR_PATH = os.getenv("LOGGER_DIR_PATH")
MODEL= "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"





LoggingDjango()

async def main(user_id: int, user_country_1: str, user_country_2: str | None, user_cv:str, top_n_interval: int, num_suitable_jobs: int):
	
	start_time = asyncio.get_event_loop().time()

	logging.info(f"\nStarting main().\n\nArguments.\nuser_id: {user_id}.\nuser_country_1: {user_country_1}. user_country_2: {user_country_2}\ntop_n_interval: {top_n_interval}\nnum_suitable_jobs: {num_suitable_jobs}")

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

	"""
	This function performs the following actions:

	1. Concurrently summarizes job descriptions 
	2. Tracks the cost of each summary.
	2. Constructs a message for GPT, including ids, job summaries and user CV text, while respecting a token budget.
	
	Returns a tuple containing the constructed message and a list of job summaries
	"""

	formatted_message, job_summaries = await async_format_top_jobs_summarize(
															user_id,
															user_cv,
															df,
															summarize_gpt_model="gpt-3.5-turbo-1106",
															classify_gpt_model="gpt-4"
														)


	#TODO: Modify. Job summaries need to go in postgre
	#df_summaries = pd.DataFrame(job_summaries)
	#append_parquet(df_summaries, 'summaries')
	
	gpt4_response = await async_classify_jobs_gpt_4(
												user_cv,
												formatted_message,
												classify_gpt_model = "gpt-4",
												log_gpt_messages= True
											)

	logging.info(gpt4_response)
	
	#This is just to see. We are probs not gonna use df
	#df = pd.DataFrame(matching_embeddings, columns=["id", "job_info", "timestamp", "embedding"])
	#print(rows)

	# Close the database connection
	conn.commit()
	cursor.close()
	conn.close()

	#print(df, df.info())
	print()

	elapsed_time = asyncio.get_event_loop().time() - start_time

if __name__ == "__main__":
	asyncio.run(main(user_id=40, user_country_1="India", user_country_2=None, user_cv=cv, top_n_interval=4, num_suitable_jobs=1))
	#asyncio.run(main(top_n_interval=4, num_suitable_jobs=1))
