import json
import pandas as pd
import os

import time

from dotenv import load_dotenv


load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

# -- Now we can get to it
from openai import OpenAI


def ClassificationJson(data,filename,categories):
    """
    Converts a given dataset into a JSON Lines (JSONL) file suitable for OpenAI's GPT-3.5 turbo model.
    
    Args:
        data (DataFrame or similar data structure): Input data containing text and labels.

    The function processes the input data row by row, constructing conversations for each row with a system message, user message, and an assistant message. It then writes the generated conversation data to a JSONL file.
 
    """
    # Initialize an empty list to store conversation data
    message_list = []

    # Iterate through the rows in the input data
    for _, row in data.iterrows():
        # Create a system message as an initial instruction
        system_message = {
            "role": "system",
            "content": f"given the following text: find the category in: {categories} that is most closely associated with it. Return only the category name only in following format"
        }

        # Append the system message to the conversation
        message_list.append({"messages": [system_message]})

        # Create a user message based on the 'text' column from the data
        user_message = {
            "role": "user",
            "content": f"{row['question']}"
        }

        # Append the user message to the conversation
        message_list[-1]["messages"].append(user_message)

        # Create an assistant message based on the 'coarse_label' column from the data
        assistant_message = {
            "role": 'assistant',
            "content": row['actual']
        }

        # Append the assistant message to the conversation
        message_list[-1]["messages"].append(assistant_message)

    # Write the conversation data to a JSON Lines (JSONL) file
    with open(filename, "w") as json_file:
        for message in message_list:
            # Serialize the conversation data to JSON and write it to the file
            json.dump(message, json_file)
            json_file.write("\n")


def fine_tune_model(model_id,pandas_df):
    df = pandas_df
    filename = f'ft_model.jsonl'
    text_to_openai_json(df, filename)
    loader = client.files.create(file=open(filename, "rb"), purpose='fine-tune')
    fine_tuning_job = client.fine_tuning.jobs.create(training_file=loader.id, model="gpt-3.5-turbo-1106")
    return fine_tuning_job.id


def wait_for_fine_tuning(job_id):
    while True:
        response = client.fine_tuning.jobs.retrieve(job_id)
        print(response.fine_tuned_model)
        #print(response["fine_tuned_model"])
        if response.fine_tuned_model:
            print(response.fine_tuned_model)
            return response.fine_tuned_model
        time.sleep(30)
    
