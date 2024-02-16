import pytest
import os
import sys


current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
print(grandparent_dir)
sys.path.append(grandparent_dir)

from unittest.mock import patch, MagicMock
from util.predictions.answer import generate_answers

def test_question_answering_private():
    question = "What is the capital of France?"
    context = "The capital of France is Paris."

    response = generate_answers(
        question=question,
        context=context,
        private=True
    )

    print(response)
    assert response is not None



def test_question_answering_claude():
    question = "What is the capital of France?"
    context = "The capital of France is Paris."

    response = generate_answers(
        question=question,
        context=context,
        model_type="claude",
        private=True
    )

    print(response)
    assert response is not None


def test_question_answering_gpt():
    question = "What is the capital of France?"
    context = "The capital of France is Paris."

    response = generate_answers(
        question=question,
        context=context,
        model_type="gpt",
        private=True
    )

    print(response)
    assert response is not None