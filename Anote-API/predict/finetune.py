import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def fine_tune_model(train_dataset, eval_dataset, model_name, output_dir, num_train_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(train_dataset.label_list))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_dir=output_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)