# ZERO SHOT NAMED ENTITY RECOGNITION
# EXAMPLE STYLE: https://spacy.io/universe/project/conciseconcepts

from anote import predict
import pandas as pd

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"

entities_df = pd.read_csv("./data/ner_taxonomy.csv")

ENTITIES = entities_df["PII type"].values()

"""
Entities: ["Full Name", "Home Address", "Email Address", "Social Security Number"]
"""

df = pd.read_csv("./data/ner_text.csv")

PREDICTIONS = []
for i, row in df.iterrows():
    PREDICTIONS.append(predict(
        entities=ENTITIES,
        text=row,
    )
    )
print(PREDICTIONS)
"""
Output:
My name is John Doe (Full Name) and my email is johndoe@example.com (Email Address).
I live at 123 Main St, Anytown, USA (Home Address).
My phone number is 555-123-4567 (Phone Number).
"""