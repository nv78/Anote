# evaluate.py

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from nltk.metrics import masi_distance  # For cosine similarity
from nltk.translate.bleu_score import sentence_bleu  # For BLEU score
from nltk.translate.bleu_score import SmoothingFunction  # For BLEU score smoothing
from transformers import pipeline  # For LLM evaluation
from rouge_score import rouge_scorer  # For ROUGE score

# Evaluation function for classification models
def evaluate_classification_model(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred)
    return {
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "classification_report": report,
    }

# Evaluation function for question-answering models using cosine similarity
def evaluate_question_answering_cosine_similarity(reference, response):
    similarity = masi_distance(set(reference.split()), set(response.split()))
    return {"cosine_similarity": similarity}

# Evaluation function for question-answering models using BLEU score
def evaluate_question_answering_bleu_score(reference, response):
    reference_tokens = reference.split()
    response_tokens = response.split()
    bleu_score = sentence_bleu(
        [reference_tokens], response_tokens, smoothing_function=SmoothingFunction().method4
    )
    return {"bleu_score": bleu_score}

# Evaluation function for question-answering models using LLM (Language Model) evaluation
def evaluate_question_answering_llm(reference, response):
    qa_pipeline = pipeline(
        "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    result = qa_pipeline(context=reference, question=response)
    answer = result['answer']
    return {"llm_answer": answer}

# Evaluation function for question-answering models using ROUGE score
def evaluate_question_answering_rouge_score(reference, response):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, response)
    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_l": scores["rougeL"].fmeasure,
    }
