import os
import sys
from openai import OpenAI
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)


def classify_document(categories, input_data, model_type="gpt3.5_turbo", private=False):
    # Initialize OpenAI client with your API key

    if model_type == "gpt3.5_turbo":
        pred = []
        system_content = f"given the following text: find the category in: {categories} that is most closely associated with it. Return only the category name only in following format"

        for text in input_data:
            completion = client.chat.completions.create(
                model='gpt-3.5-turbo-0301',
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text}]
            )

            pred.append(completion.choices[0].message.content)

        return pred

    elif model_type == "bart":
        pipe = pipeline(model="facebook/bart-large-mnli")

        print(input_data,categories)
        completion = pipe(input_data, candidate_labels=categories)
        result = []
        for i in range(len(completion)):
            scores = completion[i]['scores']
            max_score_index = scores.index(max(scores))
            result.append(completion[i]['labels'][max_score_index])

        return result

    else:
        raise ValueError("Invalid model_type. Supported types: setfit, bert, gpt3.5 turbo")