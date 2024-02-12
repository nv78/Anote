import pytest
from anote import predict

@pytest.mark.parametrize("review, expected_sentiment", [
    ("This movie was absolutely fantastic! Highly recommended.", "positive"),
    ("I was disappointed with the plot and acting.", "negative"),
    ("The cinematography in this film is breathtaking.", "positive"),
    ("Mediocre movie, nothing special.", "neutral"),
    ("The performances were outstanding.", "positive"),
])
def test_sentiment_analysis(review, expected_sentiment):
    API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"
    CATEGORIES = ["positive", "negative", "neutral"]

    actual_sentiment = predict(categories=CATEGORIES, text=review)

    assert actual_sentiment == expected_sentiment


def test_bert():

    predict(modeltype = "BERT", categories="", text="")
    pass