from anote import upload, predict
import glob

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"

directory_path = '/data/10-KS/'

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

all_text = " ".join(DOCUMENTS)

EXAMPLES = [
  {
    "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.",
    "answer": "$1577.00"
  },
  {
    "question": "Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.",
    "answer": "$8.70"
  },
  {
    "question": "Is 3M a capital-intensive business based on FY2022 data?",
    "answer": "No, the company is managing its CAPEX and Fixed Assets pretty efficiently, which is evident from below key metrics: CAPEX/Revenue Ratio: 5.1 percent Fixed assets/Total Assets: 20% Return on Assets= 12.4%"
  },
   {
    "question": "What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.",
    "answer": "Operating Margin for 3M in FY2022 has decreased by 1.7% primarily due to: -Decrease in gross Margin -mostly one-off charges including Combat Arms Earplugs litigation, impairment related to exiting PFAS manufacturing, costs related to exiting Russia and divestiture-related restructuring charges"
  }
]

QUESTIONS = [
    "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.",
    "You are an investment banker and your only resource(s) to answer the following question is (are): the statement of financial position and the cash flow statement. Here's the question: what is the FY2015 operating cash flow ratio for Adobe? Operating cash flow ratio is defined as: cash from operations / total current liabilities. Round your answer to two decimal places",
    "How much has the effective tax rate of American Express changed between FY2021 and FY2022?"
]

PREDICTIONS = []
for question in QUESTIONS:
    PREDICTIONS.append(predict(
        questions=question,
        examples=EXAMPLES,
        text=str(all_text)
    ))
print(PREDICTIONS)
"""
Output:
[
    "24.26",
    "0.66",
    "The effective tax rate for American Express has changed/dropped from 24.6% in FY 2021 to 21.6% in FY 2022."
]
"""


