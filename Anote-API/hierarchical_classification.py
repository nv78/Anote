# ZERO SHOT HIERARCHICAL CLASSIFICATION

from anote import predict
import pandas as pd

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"

categories_df = pd.read_csv("./data/hierarchical_taxonomy.csv")

CATEGORIES = []

for i, row in categories_df.iterrows():
    category = row['Category']
    subcategory = row['Sub-Category']
    subsubcategory = row['Sub-Sub-Category']

    category_item = {
        'Category': category,
        'Sub-Category': subcategory,
        'Sub-Sub-Category': subsubcategory
    }

    CATEGORIES.append(category_item)

print(CATEGORIES)
"""
Categories:
[
    {'Category': 'Clothing', 'Sub-Category': 'Shoes', 'Sub-Sub-Category': 'Cleats'},
    {'Category': 'Clothing', 'Sub-Category': 'Shoes', 'Sub-Sub-Category': 'Sandals'},
    {'Category': 'Electronics', 'Sub-Category': 'Computers', 'Sub-Sub-Category': 'Laptops'},
    {'Category': 'Electronics', 'Sub-Category': 'Computers', 'Sub-Sub-Category': 'Desktops'}
]
"""

df = pd.read_csv("./data/hierarchical_text.csv")
TEXT = [row for i, row in df.iterrows()]

print(TEXT)
"""
Text:
[
    "Great product, really enjoyed using it.",
    "The quality was poor and it broke after a few uses.",
    "Fast shipping, item as described. Would buy again.",
    "Not as described. Very disappointed.",
    "Excellent customer service."
]
"""

PREDICTIONS = []
for row in TEXT:
    PREDICTIONS.append(predict(
        categories=CATEGORIES,
        text=row,
        hierarchical=True
    ))

print(PREDICTIONS)
"""
Output:
[
    ['Clothing', 'Outerwear', 'Jackets'],
    ['Electronics', 'Computers', 'Laptops'],
    ['Clothing', 'Shoes', 'Sneakers'],
    ['Clothing', 'Outerwear', 'Coats'],
    ['Electronics', 'Audio', 'Headphones']
]
"""