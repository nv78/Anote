from anote import predict

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"

CATEGORIES = [
    "positive",
    "negative",
    "neutral"
]

REVIEWS = [
    "This movie was absolutely fantastic! Highly recommended.",
    "I was disappointed with the plot and acting.",
    "The cinematography in this film is breathtaking.",
    "Mediocre movie, nothing special.",
    "The performances were outstanding."
]

PREDICTIONS = []
for review in REVIEWS:
    PREDICTIONS = predict(
        categories=CATEGORIES,
        text=review
    )

print(PREDICTIONS)
"""
Output:
positive
negative
positive
neutral
positive
"""