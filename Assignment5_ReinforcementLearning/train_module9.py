from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import os

MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "models/gpt2-qa"


def format_example(example):
    q = example["question"].strip()
    a = example["answer"].strip()
    example["text"] = f"Question: {q}\nAnswer: {a}"
    return example


def main():
    #load dataset from JSONL files
    data_files = {
        "train": "data/qa_train.jsonl",
        "validation": "data/qa_valid.jsonl",
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    #add a single "text" field
    raw_datasets = raw_datasets.map(format_example)

    #loading tokenizer & model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    #tokenize
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    #data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    #simple training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=10,
        fp16=False, 
        save_total_limit=2,
    )

    #trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    #training the model
    trainer.train()

    #saving fine-tuned model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
