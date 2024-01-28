from anote import predict
import pandas as pd

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"

df = pd.read_csv("./data/text_classification.csv")

CATEGORIES = df["Category"].unique()
"""
CATEGORIES: ["Bug", "Feature", "Task", "Improvement", "Documentation"]
"""
EXAMPLES = {}
for category in CATEGORIES:
    EXAMPLES[category] = df[df["Category"] == category]["Text"].tolist()

"""
EXAMPLES = {
    "Bug": [
        "Fix login page validation issue",
        "Fix broken links on the homepage",
        "Fix formatting issue in the reports module",
        "Fix performance degradation issue",
        "Fix permission issue in the admin panel"
    ],
    "Feature": [
        "Add search functionality to the website",
        "Implement email notification system",
        "Implement file upload functionality",
        "Implement multi-language support",
        "Implement social media sharing feature"
    ],
    "Task": [
        "Implement user profile management module",
        "Create API documentation for integration",
        "Create automated test scripts",
        "Create user management module",
        "Create automated build process"
    ],
    "Improvement": [
        "Optimize database queries for better performance",
        "Optimize memory usage for better scalability",
        "Improve user interface design",
        "Improve error handling mechanism",
        "Improve search functionality speed"
    ],
    "Documentation": [
        "Update user guide documentation",
        "Write release notes for version 1.2",
        "Update installation guide documentation",
        "Update API reference documentation",
        "Update user manual documentation"
    ]
}
"""

TEST_EXAMPLES = [
    "I encountered a bug while trying to log in. It gives an error message.",
    "I suggest adding a search functionality to our website to improve user experience.",
    "The user guide documentation needs an update with the latest features."
    "We need to implement a user profile management module for our application.",
    "The database queries are slow, and we should optimize them for better performance.",
]

PREDICTIONS = []
for row in TEST_EXAMPLES:
    PREDICTIONS = predict(
        categories=CATEGORIES,
        text=row
    )

print(PREDICTIONS)
"""
Output: ['Bug', 'Feature',  'Documentation', 'Task', 'Improvement']
"""