

import pytest
import os
import sys

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(grandparent_dir)

from util.predictions.classify import classify_document
import pytest

@pytest.mark.parametrize("review, expected_sentiment", [
    ("This movie was absolutely fantastic! Highly recommended.", "positive"),
    ("I was disappointed with the plot and acting.", "negative"),
    ("The cinematography in this film is breathtaking.", "positive"),
    ("Mediocre movie, nothing special.", "neutral"),
    ("The performances were outstanding.", "positive"),
])
def test_sentiment_analysis(review, expected_sentiment):
    # Assuming predict is a function you've defined or imported elsewhere
    actual_sentiment = classify_document(task_type='classify', categories=["positive", "negative", "neutral"], input_data=review)

    assert actual_sentiment == expected_sentiment, f"Expected {expected_sentiment}, got {actual_sentiment}"