# Anote

## Overview

<img width="546" alt="Screen Shot 2022-12-21 at 8 27 33 AM" src="https://user-images.githubusercontent.com/42753347/208916243-c9aebd4d-b494-4e15-9f60-801cec636ccd.png">

Anote is an AI assisted data labeling tool. The core idea behind the Anote product is that as a user, after you label a few data points of text data, we can label the rest. This saves users time and money, while providing users accurate, high quality labeled training data to enable the successful completion of AI projects. To do this, Anote leverages novel technological breakthroughs in NLP and few shot learning to:

- **Label unstructured text data in a faster, cheaper, and more accurate way.**
- **Identify and fix mislabels in structured datasets**
- **Access new data sources to solve real world AI problems**

## Upload

<img width="540" alt="Screen Shot 2022-12-21 at 8 34 03 AM" src="https://user-images.githubusercontent.com/42753347/208917519-68014198-7b53-46b7-b289-50e45ced1d97.png">

Once you log in, the first step is to upload a new text based dataset. This can be via:

- **Uploading a Unstructured Dataset** - We support PDFs, DOCX, PPTXs, and many other file types.
- **Uploading a Structured Dataset** - We support labeled, tabular data in a CSV
- **Connecting to Datasets** - We support connections to a variety of APIs, such as Twitter and Reddit
- **Scrape Datasets** - We can scrape any sort of HTML data from websites such as LinkedIn and Wikipedia

If you don’t have a dataset readily available you can select a preloaded dataset from our datasets hub. These datasets are passed through our decomposer, which converts the raw file into rows of text data

## Customize

<img width="543" alt="Screen Shot 2022-12-21 at 8 34 47 AM" src="https://user-images.githubusercontent.com/42753347/208917623-4b8addda-2671-4d0f-8292-d5128ae845a3.png">

Once you have uploaded your dataset or selected a preloaded dataset, the next step is customizing. Here you can choose the task. 

- **Common Tasks Include** - Text classification, document labeling, identifying mislabels, named entity recognition, dialogue systems for conversational AI, co-referencing, part of speech tagging and optical character recognition, and named entity linkage

For this example, let’s say we are doing text classification. This is the point where subject matter experts come in and incorporate their intrinsic expertise into our pretrained transformer model. They can:

- **Choose their classes** - Add, modify or delete different categories of classes to fit your use case
- **Choose programmatic labeling functions** - Add Programmatic labeling functions to fit their use case.

#### Programmatic Labeling Functions

The idea behind programmatic labeling functions is that for each row of data, we check whether the row satisfies a heuristic - if so we assign it a label. Programmatic Labeling Functions can be simple heuristics such as:

- **keyword matches:** *If keyword "meow" AND NOT keyword "woof"* then Category Cat
- **multiple keyword matches:** *If keyword "the cat meows"* then Category Cat
- **named entity recognitions:** *If ENTITY PERSON* then Category Human. We support over 50 custom made entities.
- **regex expressions:** *If $WEIGHT > 150 LBS* then Category Overweight.
- **part of speech tagging:** *If keyword "run" is a VERB* then Category Exercise.

These programmatic labeling functions could also be more complex ontologies that the subject matter expert recommends, such as co-referencing and 2D Embeddings. We don’t believe that programmatic labeling functions substitute for manual data labeling, but think they serve as a solid foundation for initializing a pre-trained transformer model, like BERT for instance. 

## Annotate

<img width="543" alt="Screen Shot 2022-12-21 at 8 35 13 AM" src="https://user-images.githubusercontent.com/42753347/208917700-263082d6-479f-4891-b0f1-c493d814f884.png">

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

<img width="544" alt="Screen Shot 2022-12-21 at 8 35 41 AM" src="https://user-images.githubusercontent.com/42753347/208917838-24d11961-0abe-48c7-911b-2c89dab6d86a.png">


When you are satisfied with your data labels, you can view specific model metrics as well as download your high quality, labeled training dataset. We use a state of the art decomposer to convert all sorts of unstructured data into spreadsheets, with a column for the predicted label. You can download a CSV of your PDF, DOCX, EMAIL, HTML, SOCIAL MEDIA, or PPTX file.

#### OCR
Many documents have different formats, be it unstructured data, structured data, or semi-structured data. Documents contain tables, handwriting, buttons, checkboxes, and each document may contain key information that a user would like to extract. We leverage OCR capabilities to convert many of these raw documents into a CSV, with labels in required areas.

#### Identify and Fix Mislabels

If your data already has labels, we provide a dashboard to analyze the labels in your dataset
- **View Model Metrics** - we provide a plot of accuracy vs. the number of labels annotated by the data labeler, for a variety of different models. We also render a confusion matrix and a classification report with metrics like precision and recall.
- **Analyze Mislabels** - after a few annotations, we return a list of potentially mislabeled data points in your structured dataset, sorted by ambiguity. A user can fix these mislabel if our predictions are indeed incorrect.

## About

<img width="550" alt="Screen Shot 2022-12-21 at 8 35 58 AM" src="https://user-images.githubusercontent.com/42753347/208917894-2ed3bb42-c0fe-4d2f-a2dd-c68f81bdb196.png">

In the world of data centric AI, data labeling and modeling are often done in conjunction. Oftentimes, your work with Anote may just be the start of your AI and modeling and journey, and as partners we would love to be there for the ride. After the data annotations are complete, we are there to help with model development, and to help you revisit your data label to iterate on your model and improve performance.  For assistance with specific business use cases, and for product concept feedback and suggestions, please contact the Anote team at vidranatan@gmail.com. Below are a few of the common questions we get from customers:

#### What problem are you solving?

Anote is tackling a huge problem that can not be solved in a massively untapped market. For context, **100%** of AI projects need labeled training data to succeed. Right now, less than **0.001%** of data that is required to be labeled is labeled today, resulting in **90%** AI projects failing. Data Scientists realize this problem, as they currently spend **80%** of their time curating, cleaning, preprocessing and labeling their datasets, while only spending **20%** of their time building ML models. Right now, when companies need to label their data for AI projects, they may currently:

- **Leave their dataset unlabeled,** resulting in the project failing from the onset
- **Label their data themselves in a spreadsheet,** which is extremely tedious, time consuming and mundane work
- **Build internal tooling to help label their data,** which oftentimes leads to failure as it usually takes years to build
- **Send their dataset to a team of manual data annotators to label,** with workforces of millions of data labelers around the globe

#### What is the manual data annotation process like?

If companies decide to pay a team of manual data annotators to label their data, this can be extremely tedious work that takes the data annotators a lot of time (many months to years) to return labeled data. When companies get their data back from the data labelers, oftentimes they are not happy with the initial results, as they realize that there exist all of these label errors in their dataset. This usually results in a back-and-forth dialogue where companies ask the data labelers to fix the label errors, add more data to the dataset, and add more classes/categories of data as a result of changing business requirements. To make these adjustments, the data annotators need to manually label all of the data again.

At this point, companies have wasted months to years of their time to get high quality labeled data, which has been a massive pain point. Many companies spend high millions to low billions of dollars on data labeling, but have not seen the dividends pay off. Companies wish there was a better, faster and cheaper way to get labeled data for their AI project, but unfortunately without a high quality, massive labeled training dataset, their AI project can not succeed. To recap, manually labeling training data is not ideal for a variety of reasons, including:

- **Time** - Manually labeling data is painfully slow, and can take months to years to label.
- **Cost** - Labeling data eats up a significant portion of the AI development budget. Manually labeling data is inherently expensive, as cost normally scales linearly.
- **Accuracy** - Just because a dataset is labeled manually does not mean that the labels are correct. In fact, most well-known hand-labeled datasets have around 15% label error rates.
- **Changing Business Requirements** - As we label data, new data often comes in that is needed to be annotated, and new classes of data are added due to changing business requirements. This often requires manually relabeling all of the data again, from scratch.

#### What is unique about the Anote solution?
One thing that makes our product unique are the novel, proprietary few shot learning algorithms we are developing on the backend. For some context, a few years ago a transformer model called GPT-3 came out, which showed promise in few shot learning, but ended up being way too big, too expensive, and too slow to actually incorporate into your product with a reasonable runtime. Only a few months ago, the product that we wanted to build became possible with the advent of the Setfit model. This model specializes in Few Shot Learning, using a new way of doing AI called a Siamese Network. This model is way cheaper, more accurate, smaller, and faster than existing models, enabling it to actually be distilled into a product with a reasonable run time. The premise of this model is that after labeling a few data points, the rest are predicted.

Anote is the first company to take a derivative of this model (a proprietary ensemble model), bundled up with a lot of additional features, and incorporate it into a real product. We built the product, and you can use it to label specific types of unstructured text data faster and more accurately than any existing tool out there. To be clear, we do not think that the Setfit model, or our proprietary model, is the end-all-be-all model for few shot learning, but just the start of a revolution towards smaller, cheaper, and more lightweight transformer models that can actually be distilled into synchronous products to provide accurate labels with just a few data points. Our product is the first that is:

- **Synchronous** -  you get your data labeled in real time, rather than waiting for the output from data labelers
- **Few Shot** - label a few, and you label the rest, using state of the art transformer models
- **Human In The Loop** - we actively learn from the input of subject matter experts, who provide input into our model
- **Programmatic** - heuristics such as key words, entities and regex expressions are fed into our initialized model
- **Contextual** - we use LLMs to extract the context of words in sentences, not just the individual words themselves
