from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import os
import sys
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
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


def query_model(vectordb, question):
    """Asks a question based on the contents of a vector database.

    Args:
        vectordb: The vector database created from documents
        question: Question to ask the model

    Returns:
        answer: The answer given by the fine-tuned GPT model
    """

    # Assume vectordb is the vector database instance returned by the modified create_knowledge_hub function
    source1 = vectordb.similarity_search(question, k=2)[0].page_content
    source2 = vectordb.similarity_search(question, k=2)[1].page_content

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.float32,
                                                 use_auth_token=True,
                                                 load_in_8bit=False)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    max_new_tokens=512,
                    do_sample=True,
                    top_k=30,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id)

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.1})

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",  # Ensure this is correctly defined or adjusted
                                           retriever=retriever,
                                           return_source_documents=True)

    answer = qa_chain(question)['result']

    return answer

def rag_private(user_query, path_to_documents):
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


