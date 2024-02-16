import pytest
from unittest.mock import patch, MagicMock
from predict.answer import generate_answers

@pytest.fixture
def setup_question_answering():
    question = "What is the capital of France?"
    context = "The capital of France is Paris."
    return question, context

# Test for Custom QA Model
@patch("predict.answer.pipeline")
def test_generate_answers_custom_qa(mock_pipeline, setup_question_answering):
    question, context = setup_question_answering
    
    # Mocking the pipeline's return value
    mock_pipeline.return_value = MagicMock(return_value={"answer": "Paris"})
    
    answer = generate_answers(question, context, model_type="custom_qa")
    assert "Paris" in answer, "The function should return Paris as the answer"
