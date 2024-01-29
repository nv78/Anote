from anote import upload, predict
import glob

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"

directory_path = '/data/Resume/'

files = glob.glob(directory_path + '*')

DOCUMENTS = []

for file in files:
    text = upload(
        file=file,
        decomposition="PER_DOCUMENT"
    )
    DOCUMENTS.append(text)

PREDICTIONS = []
for document in DOCUMENTS:
    PREDICTIONS.append(predict(
        questions="summarize this document",
        text=document
    ))
print(PREDICTIONS)
"""
Output:
[
    "Dedicated and highly skilled Software Engineer with 5+ years of experience in designing, developing, and deploying robust web applications. Proficient in multiple programming languages and frameworks, including Python, JavaScript, and React. Proven track record of leading cross-functional teams and delivering projects on time and within budget. Passionate about solving complex technical challenges and continuously learning to stay updated with the latest industry trends.",
    "Results-driven Marketing Manager with a strong background in digital marketing and brand management. Over 8 years of experience creating and implementing data-driven marketing strategies that have consistently increased brand visibility and revenue. Expertise in SEO, SEM, email marketing, and social media campaigns. Proven ability to lead and mentor marketing teams to exceed targets. A creative thinker with a strong analytical mindset, dedicated to delivering measurable results."
    "Compassionate and highly skilled Registered Nurse with 10+ years of experience in providing exceptional patient care in various healthcare settings. Proficient in critical care, emergency medicine, and patient assessment. Strong interpersonal and communication skills with a focus on patient education and advocacy. Proven ability to work effectively in high-pressure environments, making sound clinical decisions and providing comprehensive healthcare services. Dedicated to improving patient outcomes and continuously advancing nursing skills through ongoing education and certifications."
]
"""