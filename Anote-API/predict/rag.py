import torch
from transformers import AutoTokenizer, RagTokenizer, RagRetriever, RagTokenForGeneration

def initialize_rag_model(model_name, retriever_name):
    # Initialize the RAG model with a pre-trained model and retriever
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    retriever = RagRetriever.from_pretrained(retriever_name)
    model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever)

    return model, tokenizer

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

    # Generate an answer using RAG
    generated_ids = rag_model.generate(
        input_ids=input_dict["input_ids"],
        max_length=50,  # Adjust the max length as needed
        num_beams=4,   # Adjust the number of beams for beam search
    )

    # Decode and return the answer
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer