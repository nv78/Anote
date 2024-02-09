# test_evaluate.py

import pytest
from evaluate import (
    evaluate_classification_model,
    evaluate_question_answering_cosine_similarity,
    evaluate_question_answering_bleu_score,
    evaluate_question_answering_llm,
    evaluate_question_answering_rouge_score,
)

# Define fixtures for test data
@pytest.fixture
def classification_data():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 1, 0, 0, 1]
    return y_true, y_pred

@pytest.fixture
def qa_data():
    reference_text = "The capital of France is Paris."
    response_text = "Paris is the capital of France."
    return reference_text, response_text

# Test cases for classification evaluation
class TestClassificationEvaluation:
    def test_accuracy(self, classification_data):
        y_true, y_pred = classification_data
        result = evaluate_classification_model(y_true, y_pred)
        assert "accuracy" in result
        assert result["accuracy"] == 0.6  # Adjust the expected value as needed

    def test_precision(self, classification_data):
        y_true, y_pred = classification_data
        result = evaluate_classification_model(y_true, y_pred)
        assert "precision" in result
        assert result["precision"] == 0.6  # Adjust the expected value as needed

# Test cases for question-answering evaluation
class TestQuestionAnsweringEvaluation:
    def test_cosine_similarity(self, qa_data):
        reference_text, response_text = qa_data
        result = evaluate_question_answering_cosine_similarity(reference_text, response_text)
        assert "cosine_similarity" in result
        assert result["cosine_similarity"] == 1.0  # Adjust the expected value as needed

    def test_bleu_score(self, qa_data):
        reference_text, response_text = qa_data
        result = evaluate_question_answering_bleu_score(reference_text, response_text)
        assert "bleu_score" in result
        assert result["bleu_score"] == 1.0  # Adjust the expected value as needed

    def test_llm_answer(self, qa_data):
        reference_text, response_text = qa_data
        result = evaluate_question_answering_llm(reference_text, response_text)
        assert "llm_answer" in result
        assert result["llm_answer"] == "Paris"  # Adjust the expected value as needed

    def test_rouge_score(self, qa_data):
        reference_text, response_text = qa_data
        result = evaluate_question_answering_rouge_score(reference_text, response_text)
        assert "rouge_1" in result
        assert result["rouge_1"] == 1.0  # Adjust the expected value as needed
        assert "rouge_2" in result
        assert result["rouge_2"] == 1.0  # Adjust the expected value as needed
        assert "rouge_l" in result
        assert result["rouge_l"] == 1.0  # Adjust the expected value as needed
