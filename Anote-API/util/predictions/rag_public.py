from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import os
import sys
from langchain.document_loaders import PyPDFLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import Chroma
from torch import cuda
import torch
import pandas as pd

import os
from pathlib import Path
import torch as cuda  # Assuming cuda from torch is being used for GPU support


def create_knowledge_hub(documents_directory_path):
    """
    Args:
        documents_directory_path: Path to the directory containing all documents to be processed
    Returns:
        vectordb: The vector database with the information from all documents
        db_directory: The path to the vector database
    """
    
    # Define a fixed database directory name
    db_directory = "db_knowledge_hub"

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    # Check if the database directory already exists
    if os.path.exists(db_directory):
        print("Using existing database for the knowledge hub.")
        vectordb = Chroma(persist_directory=db_directory, embedding_function=embed_model)
    else:
        print("Creating new database for the knowledge hub.")
        documents = []

        # Loop through each document in the directory and load them
        for file_name in os.listdir(documents_directory_path):
            document_path = os.path.join(documents_directory_path, file_name)
            if os.path.isfile(document_path):
                loader = PyPDFLoader(document_path)
                documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1300,
            chunk_overlap=5,
            separators=["\n\n", "\n", " ", ""],
            length_function=len)
        texts = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embed_model,
            persist_directory=db_directory
        )
        vectordb.persist()

    return vectordb, db_directory


import openai

# Set your OpenAI API key here
openai.api_key = 'your_openai_api_key_here'

def query_model(vectordb, question):
    """Asks a question based on the contents of a vector database using OpenAI's GPT-4.

    Args:
        vectordb: The vector database created from documents
        question: Question to ask the model

    Returns:
        answer: The answer given by the GPT-4 model
    """

    # Fetch the most relevant documents as context
    source1 = vectordb.similarity_search(question, k=2)[0].page_content
    source2 = vectordb.similarity_search(question, k=2)[1].page_content

    # Prepare the prompt with the relevant documents
    prompt = f"Based on the following information:\n\n{source1}\n\nAnd:\n\n{source2}\n\nQuestion: {question}\nAnswer:"

    # Use OpenAI API to query GPT-4
    response = openai.Completion.create(
        model="gpt-4",  # Replace with the correct GPT-4 model identifier when available
        prompt=prompt,
        temperature=0.7,
        max_tokens=150,
        n=1,
        stop=None,
        top_p=1.0
    )

    # Extract the answer from the response
    answer = response.choices[0].text.strip()

    return answer


def rag_public(user_query, path_to_documents):
    """
    Processes documents to create a knowledge hub and answers a user query based on this hub.

    Args:
        user_query: The question the user wants answered.
        path_to_documents: The directory path where documents are stored.

    Returns:
        answer: The answer to the user's query, based on the processed documents.
    """

    # Step 1: Create the knowledge hub from documents
    vectordb, db_directory = create_knowledge_hub(path_to_documents)

    # Step 2: Query the model to get an answer based on the created knowledge hub
    answer = query_model(vectordb, user_query)

    return answer