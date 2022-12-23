# Anote

## Overview

<img width="803" alt="Screen Shot 2022-12-23 at 2 18 21 PM" src="https://user-images.githubusercontent.com/42753347/209396598-1c31fc45-9993-4ac9-bfce-6407d396d61a.png">

Anote is an AI assisted data labeling tool. The core idea behind the Anote product is that as a user, after you label a few data points of text data, we can label the rest. This saves users time and money, while providing users accurate, high quality labeled training data to enable the successful completion of AI projects. To do this, Anote leverages novel technological breakthroughs in NLP and few shot learning to:

- **Label unstructured text data in a faster, cheaper, and more accurate way.**
- **Identify and fix mislabels in structured datasets**
- **Access new data sources to solve real world AI problems**

## Upload


<img width="803" alt="Screen Shot 2022-12-23 at 2 18 35 PM" src="https://user-images.githubusercontent.com/42753347/209396638-c38fdba5-f6ce-4bcc-bca8-5b9d7890ea24.png">

Once you log in, the first step is to upload a new text based dataset. This can be via:

- **Uploading a Unstructured Dataset** - We support PDFs, DOCX, PPTXs, and many other file types.
- **Uploading a Structured Dataset** - We support labeled, tabular data in a CSV
- **Connecting to Datasets** - We support connections to a variety of APIs, such as Twitter and Reddit
- **Scrape Datasets** - We can scrape any sort of HTML data from websites such as LinkedIn and Wikipedia

If you don’t have a dataset readily available you can select a preloaded dataset from our datasets hub. These datasets are passed through our decomposer, which converts the raw file into rows of text data

## Customize

<img width="804" alt="Screen Shot 2022-12-23 at 2 18 49 PM" src="https://user-images.githubusercontent.com/42753347/209396661-ddb5e8bb-6c21-4138-bb29-65a0b67a086e.png">

Once you have uploaded your dataset or selected a preloaded dataset, the next step is customizing. Here you can choose the task. 

- **Common Tasks Include** - Text classification, document labeling, identifying mislabels, named entity recognition, dialogue systems for conversational AI, co-referencing, part of speech tagging and optical character recognition, and named entity linkage

For this example, let’s say we are doing text classification. This is the point where subject matter experts come in and incorporate their intrinsic expertise into our pretrained transformer model. They can:

- **Choose their classes** - Add, modify or delete different categories of classes to fit your use case
- **Choose programmatic labeling functions** - Add Programmatic labeling functions to fit their use case.

#### Programmatic Labeling Functions

The idea behind programmatic labeling functions is that for each row of data, we check whether the row satisfies a heuristic - if so we assign it a label. Programmatic Labeling Functions can be simple heuristics such as:

- **keyword matches:** *If keyword "meow" AND NOT keyword "woof"* then Category Cat
- **multiple keyword matches:** *If keyword "the cat meows"* then Category Cat
- **named entity recognitions:** *If ENTITY PERSON* then Category Human.
- **regex expressions:** *If $WEIGHT > 150 LBS* then Category Overweight.
- **part of speech tagging:** *If keyword "run" is a VERB* then Category Exercise.

These programmatic labeling functions could also be more complex ontologies that the subject matter expert recommends, such as co-referencing and 2D Embeddings. We don’t believe that programmatic labeling functions substitute for manual data labeling, but think they serve as a solid foundation for initializing a pre-trained transformer model, like BERT for instance. 

## Annotate

<img width="804" alt="Screen Shot 2022-12-23 at 2 18 57 PM" src="https://user-images.githubusercontent.com/42753347/209396675-fea4396a-1f06-4960-810d-7868685835ea.png">

After creating initial programmatic labeling functions, the next step is to begin annotating. We designed our annotation interface to be as seamless, easy to use and enjoyable as possible, almost like a quizlet for data labeling. The GUI is flexible, able to be modified based upon dataset, document type and task. Here, the data labeler sees a row of text data and the corresponding class prediction. The rows shown to the data labeler are sorted by uncertainty (lowest probability) or volatility (biggest impact) that way the data labeler is annotating points that will make the biggest difference to the model. The data labeler can:
- **choose** the category of the specific label
- **add** keywords that influenced the labeling decision
- **skip** the row of data if they are unsure about the data label

#### Few Shot Learning
After each row of data labeled, our product is actively learns from the data labeler, using state of the art few shot learning. The magic happens on the backend, where we have a transformer based few shot learning model continuously learning in real time from the input of the data labelers. Our model is an ensemble of smaller, cheaper, faster, and more accurate large language models that learn from the human in the loop, customized to domain specific tasks based on the dataset, document types and annotations. These are not GPT-3 like billion-parameter models, but instead are derivatives of Setfit / smaller LLMs that run on the order of seconds. For reference, Setfit was released in September 2022, and received state of the art performance in many few shot learning tasks, at a fraction of the cost and size. Our model is lightweight enough to be distilled into products.

Because of the power of our novel transformer based few shot learning model, we are able to learn the predictions of many labeled data points in a small fraction of the time. For certain domain specific tasks, our goal is to be able to accurately predict the labels of one million rows of data with just one thousand labels. This saves data annotators time with the remaining 999000 rows of data, while still incorporating their intrinsic subject matter expertise into the AI model. In other words, you label a few, we label the rest.

#### Review Annotations

After each annotation made, the labels appear in a dashboard to the users right, where a reviewer can come and view the previous data labels made. The reviewer has the option to delete and/or correct annotations if they have been done incorrectly, as well as to evaluate data labelers on performance.

## Download

<img width="805" alt="Screen Shot 2022-12-23 at 2 19 04 PM" src="https://user-images.githubusercontent.com/42753347/209396698-a425ee06-968b-47ca-897b-54a6750a80d2.png">

When you are satisfied with your data labels, you can view specific model metrics as well as download your high quality, labeled training dataset. We use a state of the art decomposer to convert all sorts of unstructured data into spreadsheets, with a column for the predicted label. You can download a CSV of your PDF, DOCX, EMAIL, HTML, SOCIAL MEDIA, or PPTX file.

#### OCR
Many documents have different formats, be it unstructured data, structured data, or semi-structured data. Documents contain tables, handwriting, buttons, checkboxes, and each document may contain key information that a user would like to extract. We leverage OCR capabilities to convert many of these raw documents into a CSV, with labels in required areas.

#### Identify and Fix Mislabels

If your data already has labels, we provide a dashboard to analyze the labels in your dataset
- **View Model Metrics** - we provide a plot of accuracy vs. the number of labels annotated by the data labeler, for a variety of different models. We also render a confusion matrix and a classification report with metrics like precision and recall.
- **Analyze Mislabels** - after a few annotations, we return a list of potentially mislabeled data points in your structured dataset, sorted by ambiguity. A user can fix these mislabel if our predictions are indeed incorrect.

## About

<img width="806" alt="Screen Shot 2022-12-23 at 2 19 14 PM" src="https://user-images.githubusercontent.com/42753347/209396710-b8a2997e-3f54-463b-be13-ebfb36d99c68.png">

In the world of data centric AI, data labeling and modeling are often done in conjunction. Oftentimes, your work with Anote may just be the start of your AI and modeling and journey, and as partners we would love to be there for the ride. After the data annotations are complete, we are there to help with model development, and to help you revisit your data label to iterate on your model and improve performance.  For assistance with specific business use cases, and for product feedback, please contact the Anote team at vidranatan@gmail.com.
