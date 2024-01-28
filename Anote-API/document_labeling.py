from anote import upload, predict
import glob

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"
directory_path = '/data/doclabeling/'

files = glob.glob(directory_path + '*')

CATEGORIES = [
    "Contracts",
    "Regulatory",
    "Litigation",
    "Legal Opinions"
]

DOCUMENTS = []

for file in files:
    text = upload(
        file=file,
        decomposition="PER_DOCUMENT"
    )
    DOCUMENTS.append(text)

PREDICTIONS = []
for document in DOCUMENTS:
    PREDICTIONS = predict(
        categories=CATEGORIES,
        text=document
    )
print(PREDICTIONS)
"""
Output:
Litigation
Regulatory
Legal Opinions
"""