import pytest
import os
import sys


current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
print(grandparent_dir)
sys.path.append(grandparent_dir)

from unittest.mock import patch, MagicMock
from predictions.answer import generate_answers

def question_answering_private():
    question = "What is the capital of France?"
    context = "The capital of France is Paris."

    response = generate_answers(
        question=question,
        context=context,
        private=True
    )

    print(response)
    assert response is not None

# # Test for Custom QA Model
# @patch("predict.answer.pipeline")
# def test_generate_answers_custom_qa(mock_pipeline, setup_question_answering):
#     question, context = setup_question_answering

#     # Mocking the pipeline's return value
#     mock_pipeline.return_value = MagicMock(return_value={"answer": "Paris"})

#     answer = generate_answers(question, context, model_type="custom_qa")
#     assert "Paris" in answer, "The function should return Paris as the answer"
