import torch
from transformers import AutoTokenizer, RagTokenizer, RagRetriever, RagTokenForGeneration
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import (
    LocalFileStore,
)
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
from openai import OpenAI
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import FlareChain
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import os

def initialize_rag_model(model_name, retriever_name):
    # Initialize the RAG model with a pre-trained model and retriever
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    retriever = RagRetriever.from_pretrained(retriever_name)
    model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever)

    return model, tokenizer

def create_knowledge_hub(embedding_model='text-embedding-ada-002',data_path='txt/'):
    # Load all the transcripts stored in the data folder
    loader = DirectoryLoader(data_path, glob="**/*.txt", show_progress=True)
    docs = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    # Initialize OpenAI Embeddings
    openai_embedder = OpenAIEmbeddings(model=embedding_model)

    # Cache the embeddings for faster loadup
    cache_store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(openai_embedder, cache_store, namespace="sentence")

    # Create the vector db
    db = FAISS.from_documents(documents, cached_embedder)
    return db

def rag_pred(data, model_id):
    myllm = ChatOpenAI(temperature=0.30, model_name=model_id)

    flare = FlareChain.from_llm(
        llm=myllm,
        retriever=db.as_retriever(),
        max_generation_len=700,
        min_prob=0.15, 
        )


    syntheses = []
    for index, row in data.iterrows():
        result = flare.run(row['question'])
        print(index,result)
        syntheses.append(result)

    return syntheses

def answer_question_with_rag_gpt(question, model_id, context=None):
    # Initialize the knowledge hub
    db = create_knowledge_hub(embedding_model='text-embedding-ada-002', data_path='txt/')

    # Initialize the FlareChain with the selected GPT model
    myllm = ChatOpenAI(temperature=0.30, model_name=model_id)
    flare = FlareChain.from_llm(
        llm=myllm,
        retriever=db.as_retriever(),
        max_generation_len=700,
        min_prob=0.15,
    )

    # Generate an answer
    result = flare.run(question)

    return result


def answer_question(rag_model, tokenizer, question, context):
    # Encode the question and context
    input_dict = tokenizer(
        question,
        context,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    retriever = docsearch.as_retriever(search_kwargs={"k": 2})
    
    context = retriever.retrieve(question)  # Implement the retrieve method in your retriever

    # Encode the question and retrieved context
    input_dict = tokenizer(
        question + " " + context,  # Combine question and context
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    # Generate an answer using RAG
    generated_ids = rag_model.generate(
        input_ids=input_dict["input_ids"],
        max_length=50,  # Adjust the max length as needed
        num_beams=4,   # Adjust the number of beams for beam search
    )

    # Decode and return the answer
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer