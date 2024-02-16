

import pytest
import os
import sys

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(grandparent_dir)


from util.predictions import predict
import pytest

# Use the pytest.mark.parametrize decorator on the test function
""" @pytest.mark.parametrize("review, expected_sentiment", [
    ("This movie was absolutely fantastic! Highly recommended.", "positive"),
    ("I was disappointed with the plot and acting.", "negative"),
    ("The cinematography in this film is breathtaking.", "positive"),
    ("Mediocre movie, nothing special.", "neutral"),
    ("The performances were outstanding.", "positive"),
])

# Define the sentiment_analysis function
def sentiment_analysis(review, expected_sentiment):
    API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"
    CATEGORIES = ["positive", "negative", "neutral"]
    print('expected_sentiment:', expected_sentiment)


    actual_sentiment = predict(task_type='classify', categories=CATEGORIES, input_data=review)
    print('actual_sentiment:', actual_sentiment)

    assert actual_sentiment == expected_sentiment
    print("Test Passed!")
 """

@pytest.mark.parametrize("review, expected_sentiment", [
    ("This movie was absolutely fantastic! Highly recommended.", "positive"),
    ("I was disappointed with the plot and acting.", "negative"),
    ("The cinematography in this film is breathtaking.", "positive"),
    ("Mediocre movie, nothing special.", "neutral"),
    ("The performances were outstanding.", "positive"),
])
def test_sentiment_analysis(review, expected_sentiment):
    # Assuming predict is a function you've defined or imported elsewhere
    actual_sentiment = predict(task_type='classify', categories=["positive", "negative", "neutral"], input_data=review)

    assert actual_sentiment == expected_sentiment, f"Expected {expected_sentiment}, got {actual_sentiment}"