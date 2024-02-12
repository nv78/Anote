
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

from openai import OpenAI

client = OpenAI()

def classify_document(categories,input_data, model_type="gpt3.5_turbo", private=False):
    data_dict = {'text': input_data['text'][:3], 'review' : input_data['review'][:3]}
    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict(data_dict)

    if model_type == "setfit":
        # Load SetFit model and tokenizer
        model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
        trainer = SetFitTrainer(model=model,train_dataset=dataset,)
        trainer.train()
        result = model.predict(input_data['text'])

        return result

    elif model_type == "gpt3.5_turbo":
        result = []
        system_content = f"given the following text: find the category in: {categories} that is most closely associated with it. Return only the category name only in following format"
        for text in input_data['text']:
            completion = client.chat.completions.create(
                model_id = 'gpt-3.5-turbo-0301',
                messages=[
                {"role": "system", "content":system_content },
                {"role": "user", "content": text }])            
            result.append(completion)
        return result

    elif model_type == "bert":
        label_to_id = {label: idx for idx, label in enumerate(categories)}
        id_to_label = {idx: label for idx, label in enumerate(categories)}
        model = BertForSequenceClassification.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2', num_labels=len(categories), id2label=id_to_label, label2id=label_to_id)
        tokenizer = BertTokenizerFast.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
        classification_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        result = classification_pipeline(input_data['text'])

        return result

  
    else:
        raise ValueError("Invalid model_type. Supported types: setfit, bert, gpt3.5 turbo")
