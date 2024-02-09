
import sys
from classify import classify_document
from extract import extract_information
from answer import answer_question
from finetune import fine_tune_model
from rag import generate_rag_response

def predict(task_type, input_data, private=False, examples=None):
    if task_type == "classify":
        return classify_document(input_data, private=private)
    elif task_type == "extract":
        return extract_information(input_data, private=private, examples=examples)
    elif task_type == "answer":
        return answer_question(input_data, examples=examples)
    elif task_type == "finetune":
        return fine_tune_model(input_data, private=private, examples=examples)
    elif task_type == "rag":
        return generate_rag_response(input_data, examples=examples)
    else:
        raise ValueError("Invalid task_type. Supported types: classify, extract, answer, finetune, rag")
