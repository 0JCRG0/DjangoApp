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