from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import openai
import pandas as pd
import requests
import json
import datetime
import shutil
import os
from datasets import load_dataset
import time
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

def generate_answers(question,
                     context,
                     model_type="Claude",
                     private=False):
    if model_type == "question-answering":
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    elif model_type == "custom_qa":
        model_name = "your-custom-qa-model"  # Replace with your custom QA model
    else:
        raise ValueError("Invalid model_type. Supported types: Claude, GPT, Llama2, GPT4All")

    # Initialize QA pipeline with the chosen model
    qa_generator = pipeline("question-answering", model=model_name)

    # Generate answers to the question given the context
    answers = qa_generator(question=question, context=context)


    if private:
        # Perform private answer generation logic here if needed


        pass


    if model_type == 'GPT':
        openai.api_key = "Enter Your Api Key"
        system_content = "You are a factual chatbot that answers questions about 10-K documents. You only answer with answers you find in the text, no outside information."

        output = []

        for question in questions:
            completion = openai.ChatCompletion.create(
                model= model_id ,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"{question}" }
                ])

            #print(f'text: {row}')
            print(completion.choices[0].message.content)

            output.append(completion.choices[0].message.content)

    elif model_type == 'Claude':
        claude_key = "Enter API Key"

        formatted_examples = [f"<answer> {example['answer']} </answer>" for example in examples]


        anthropic = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=claude_key,
        )

        for question in questions():
            completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=700,
            prompt = (
                f"{HUMAN_PROMPT} "
                f"You are an accurate chatbot specialized in responding to queries about 10-K documents. Please only respond in the manner demonstrated by the following examples. If an explanation wasn't ask, refrain from providing one. "
                f"Based on the following examples: {formatted_examples}, please provide your answer in the same format. "
                f"please address the question: {question}. "
                f"{AI_PROMPT}")
            )

            output.append(completion.completion)

    return output