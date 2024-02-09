import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


#Components Needed For FTING BERT

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
    do_train=True,
    do_eval=True,
    num_train_epochs=7,  # Reduced number of epochs.
    per_device_train_batch_size=10,  # Reduced batch size for training.
    per_device_eval_batch_size=35,  # Reduced batch size for evaluation.
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    #fp16=True,  # Enable mixed precision training.
)



########################################################################################################################################################




class BERT():
    def __init__(self,train_dataset,eval_dataset):
        """
        Initializes a BERT-based sequence classifier.
        Args:
            train_dataset (pandas.DataFrame): Training dataset containing 'text' and 'label' columns.
            eval_dataset (pandas.DataFrame): Evaluation dataset containing 'text' and 'label' columns.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=512)    
        
        self.train_encodings = self.tokenizer(train_dataset['text'].to_list(), truncation=True, padding=True)
        self.eval_encodings = self.tokenizer(eval_dataset['text'].to_list(), truncation=True, padding=True)

        self.train_labels = train_dataset.label.to_list()
        self.eval_labels = eval_dataset.label.to_list()

        self.id_to_label = {idx: label for idx, label in enumerate(set(self.train_labels))}
        self.label_to_id = {label: idx for idx, label in enumerate(set(self.eval_labels))}

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, id2label =  self.id_to_label, label2id =  self.label_to_id)

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

        return self.ft_model, self.metrics
    

    def model_evaluation(self):
        self.acc = self.metrics['eval_Accuracy']
        self.precision = self.metrics['eval_Precision']
        self.recall = self.recall['eval_Recall']

        return self.acc , self.precision, self.recall 
   

      
    def prediction(self,texts):
        

        self.inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


        with torch.no_grad():
            self.outputs = self.fine_tune_bert(**self.inputs)

        # Get predicted label IDs
        predicted_label_ids = torch.argmax(self.outputs.logits, dim=1).tolist()

        # Convert predicted label IDs to labels
        predicted_labels = [self.outputs.config.id2label[label_id] for label_id in predicted_label_ids]

        return predicted_labels 

    


















        