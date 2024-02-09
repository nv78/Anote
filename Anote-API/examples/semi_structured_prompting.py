
from anote import upload, predict
import glob

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"
directory_path = '/data/earningscalltranscripts/'

files = glob.glob(directory_path + '*')

DOCUMENTS = []

for file in files:
    text = upload(
        file=file,
        decomposition="PER_DOCUMENT"
    )
    DOCUMENTS.append(text)

QUESTIONS = [
    "summarize this document in 3 sentences", # open ended question
    "what was the revenue in 2022", # numerical, extraction answer
    "are their any risk narratives in this document" # yes, no multiple choice question
]

EXAMPLES = [
    {
        "text": "training document 1 text",
        "questions": QUESTIONS,
        "answers": {
            QUESTIONS[0]: "This document discusses the company's financial performance, future plans, and market outlook.",
            QUESTIONS[1]: f"The revenue in 2022 was approximately $25,000,000.",
            QUESTIONS[2]: "Yes"
        }
    },
    {
        "text": "training document 2 text",
        "questions": QUESTIONS,
        "answers": {
            QUESTIONS[0]: "The document covers financial results and strategic goals.",
            QUESTIONS[1]: f"$80,000,000.",
            QUESTIONS[2]: "No"
        }
    }
]

PREDICTIONS = []
for question in QUESTIONS:
    for document in DOCUMENTS:
        PREDICTIONS.append(predict(
            questions=question,
            examples=EXAMPLES,
            text=document
        ))
print(PREDICTIONS)
"""
Output:

Predictions:
[{
    "text": "training document 3 text",
    "questions": QUESTIONS,
    "answers": {
        QUESTIONS[0]: "In this document, the company discusses its quarterly financial performance, expansion plans, and market trends.",
        QUESTIONS[1]: f"The revenue for the year 2022 reached approximately $90,000,000.",
        QUESTIONS[2]: "Yes"
    }
},{
    "text": "training document 4 text",
    "questions": QUESTIONS,
    "answers": {
        QUESTIONS[0]: "This document provides insights into the company's financial health, product innovation, and competitive landscape.",
        QUESTIONS[1]: f"The reported revenue in 2022 was approximately $75,000,000.",
        QUESTIONS[2]: "No"
    }
}]
"""