import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def fine_tune_model(train_dataset, eval_dataset, model_name, output_dir, num_train_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(train_dataset.label_list))

    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
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


    fine_tuning = SFTTrainer(
        model= model,
        train_dataset= train_dataset,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args= training_args,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    #trainer.train()

    # Fine-tune the model(the way I did with LLAMA)
    fine_tuning.train

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)