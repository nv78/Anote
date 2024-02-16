
import sys
import os

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, './predict'))
print(grandparent_dir)
sys.path.append(grandparent_dir)

from predictions.classify import classify_document
# from predictions.extract import extract_information
from predictions.answer import generate_answers
# from predictions.QAFineTune import fine_tune_model
# from predictions.rag import generate_rag_response

def predict(task_type, input_data, categories, private=False, examples=None):
    if task_type == "classify":
        return classify_document(categories,input_data, private=private)
    # elif task_type == "extract":
    #     return extract_information(input_data, private=private, examples=examples)
    elif task_type == "answer":
        return generate_answers(input_data, examples=examples)
    # elif task_type == "finetune":
    #     return fine_tune_model(input_data, private=private, examples=examples)
    # elif task_type == "rag":
    #     return generate_rag_response(input_data, examples=examples)
    else:
        raise ValueError("Invalid task_type. Supported types: classify, extract, answer, finetune, rag")
