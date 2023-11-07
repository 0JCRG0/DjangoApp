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
	
	#TODO: Timer needs to be async
	start_time = timeit.default_timer()

	logging.info(f"\nArguments.\n\nuser_id: {user_id}.\nuser_country_1: {user_country_1}. user_country_2: {user_country_2}\ntop_n_interval: {top_n_interval}\nnum_suitable_jobs: {num_suitable_jobs}")

	# create a connection to the PostgreSQL database
	conn = psycopg2.connect(LOCAL_POSTGRE_URL)
	cursor = conn.cursor()
	cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')

	#Register the vector type with your connection or cursor
	register_vector(conn)
	
	all_country_values = fetch_all_country_values(user_country_1, user_country_2)
	
	#It can be all the rows or simply the ids
	#rows = filter_pgvector_two_weeks_country(all_country_values, cursor)

	#Embed user's query
	query_embedding = e5_base_v2_query(user_cv)

	rows = testing(query_embedding, all_country_values, cursor)

	#This is just to see. We are probs not gonna use df
	#df = pd.DataFrame(rows, columns=["id", "job_info", "timestamp", "embedding"])
	print(rows)

	# Close the database connection
	conn.commit()
	cursor.close()
	conn.close()

	#print(df, df.info())
	

if __name__ == "__main__":
	asyncio.run(main(user_id=40, user_country_1="India", user_country_2=None, user_cv=cv, top_n_interval=4, num_suitable_jobs=1))
	#asyncio.run(main(top_n_interval=4, num_suitable_jobs=1))
