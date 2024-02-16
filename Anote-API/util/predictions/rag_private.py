from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import os
import sys
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
import tiktoken
import PyPDF2
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.vectorstores import Chroma
import json
import datetime
from torch import cuda
import torch
import pandas as pd

    
def create_knowledge_hub(document_path, doc_name):
    """
    Args:
        document_path: Relative path to the 10-K hosted locally on the user's computer
        doc_name: The name of the document, used to identify the vector database
    Returns:
        vectordb: The vector database with the information from the 10-K
        db_directory: The path to the vector database
    """

    # Normalize doc_name to create a valid directory name
    normalized_doc_name = doc_name.replace(' ', '_').replace('/', '_')
    db_directory = "db_" + normalized_doc_name
    
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    # Check if the database directory already exists
    if os.path.exists(db_directory):
        print(f"Using existing database for document: {doc_name}")
        # Load and return the existing database
        vectordb = Chroma(persist_directory=db_directory, embedding_function=embed_model)
    else:
        print(f"Creating new database for document: {doc_name}")

        loader = PyPDFLoader(document_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1300, 
            chunk_overlap=5,
            separators=["\n\n", "\n", " ", ""],
            length_function=len)
        texts = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=texts, 
            embedding= embed_model,  # Make sure 'embeddings' is defined or passed to the function
            persist_directory=db_directory
        )
        vectordb.persist()

    return vectordb, db_directory

def query_model(path_to_10k, doc_name, question):
    """Asks LLAMA a question based off a local 10-K document.

    Args:
        path_to_10k: Relative path to the 10-K hosted locally on the user's computer
        question: Question to ask the model

    Returns:
        answer: The answer given by the fine-tuned GPT model
    """

    db, db_dir = create_knowledge_hub(path_to_10k, doc_name)

    source1 = db.similarity_search(question, k = 2)[0].page_content
    source2 = db.similarity_search(question, k = 2)[1].page_content

    ## EDIT THIS PART
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float32,
                                             use_auth_token=True,
                                             load_in_8bit=False
                                          )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
    
    retriever = db.as_retriever(search_kwargs={"k": 2})

    pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.float,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
    
    llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.1})
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
    
    answer = qa_chain(question)['result']
    

    return answer
