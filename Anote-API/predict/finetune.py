import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# FTING BERT

class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        """
          This construct a dict that is (index position) to encoding pairs.
          Where the Encoding becomes tensor(Encoding), which is an requirements
          for training the model
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        """
        Returns the number of data items in the dataset.
        """
        return len(self.labels)



def compute_metrics(pred):
    # Extract true labels from the input object
    labels = pred.label_ids
    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)
    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro',zero_division=1)
    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

training_args = TrainingArguments(
    output_dir='./BERTModel2',
    do_train=True,
    do_eval=True,
    num_train_epochs=1,  # Reduced number of epochs.
    per_device_train_batch_size=5,  # Reduced batch size for training.
    per_device_eval_batch_size=20,  # Reduced batch size for evaluation.
    warmup_steps=50,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    fp16=True,  # Enable mixed precision training.
)


########################################################################################################################################################




class FT_BERT:
    def __init__(self,train_dataset,eval_dataset,num_labels):
        """ Initializes a BERT-based sequence classifier.

        Args:
            train_dataset (pandas.DataFrame): Training dataset containing 'text' and 'label' columns. 
                The 'label' column must contain string values.
            eval_dataset (pandas.DataFrame): Evaluation dataset containing 'text' and 'label' columns. 
                The 'label' column must contain string values.

        Example:
            ```python
            # Initialize FTING BERT classifier
            BERT = Fine_Tuning_BERT(train_df, test_df, num_labels=5)

            # Texts for prediction
            texts =  ["this product is trash"]

            # Once BERT finishes fine-tuning, you can call the .prediction method with an array of texts
            predicted_class = BERT.prediction(texts=texts)
            print(predicted_class) """


        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=512)

        self.train_encodings = self.tokenizer(train_dataset['text'].to_list(), truncation=True, padding=True)
        self.eval_encodings = self.tokenizer(eval_dataset['text'].to_list(), truncation=True, padding=True)

        self.train_labels = train_dataset.labels.to_list()
        self.eval_labels = eval_dataset.labels.to_list()


        self.train_labels = []
        for label in train_dataset.labels.to_list():
          if isinstance(label, str):
            self.train_labels.append(label)
          else:
            raise ValueError("All labels in train_dataset.labels must be strings.")

        self.eval_labels = []
        for label in eval_dataset.labels.to_list():
          if isinstance(label, str):
            self.eval_labels.append(label)
          else:
            raise ValueError("All labels in eval_dataset.labels must be strings.")


        self.all_labels = self.eval_labels + self.train_labels

        print(self.all_labels)

        self.id_to_label = {idx: label for idx, label in enumerate(set(self.all_labels))}
        self.label_to_id = {label: idx for idx, label in self.id_to_label.items()}


        self.train_labels = [self.label_to_id[label] for label in self.train_labels]

        self.eval_labels = [self.label_to_id[label] for label in self.eval_labels]

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 5 , id2label =  self.id_to_label, label2id =  self.label_to_id)



        self.train_dataloader = DataLoader(self.train_encodings,self.train_labels)
        self.eval_dataloader = DataLoader(self.eval_encodings,self.eval_labels)


        self.fine_tune_bert = self.fine_tuning()

    def fine_tuning(self):
        self.trainer = Trainer(
            #the pre-trained bert model that will be fine-tuned
            model=self.model,
            #training arguments that we defined above
            args=training_args,
            train_dataset= self.train_dataloader,
            eval_dataset = self.eval_dataloader ,
            compute_metrics= compute_metrics
        )

        self.trainer.train()

        self.ft_model = self.trainer.train()

        self.metrics = self.trainer.evaluate()

        return self.model


    def model_evaluation(self):
        self.acc = self.metrics['eval_Accuracy']
        self.precision = self.metrics['eval_Precision']
        self.recall = self.recall['eval_Recall']

        return self.acc , self.precision, self.recall



    def prediction(self,texts):
        # Text must be an array 


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        self.inputs.to(self.device)


        with torch.no_grad():
            self.outputs = self.model(**self.inputs)

        # Get predicted label IDs
        predicted_label_ids = torch.argmax(self.outputs.logits, dim=1).tolist()



        return predicted_label_ids



