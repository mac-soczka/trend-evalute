import os
import torch
import yaml
import json
from pathlib import Path
from datasets import load_dataset
from transformers.integrations import WandbCallback
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Load configuration from YAML file
def load_config(config_file="config.yml"):
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# Preprocess and tokenize dataset
def preprocess_dataset(dataset, tokenizer, max_length):
    model_max_length = tokenizer.model_max_length
    effective_max_length = min(max_length, model_max_length)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=effective_max_length
        )

    return dataset.map(tokenize_fn, batched=True)

# Evaluate the model before and after fine-tuning
def evaluate_model(model_name, dataset, max_length, fine_tuned=False):
    model_path = f"fine_tuned_models/{model_name.replace('/', '_')}" if fine_tuned else model_name
    print(f"\nEvaluating {'fine-tuned' if fine_tuned else 'pretrained'} model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_dataset = preprocess_dataset(dataset, tokenizer, max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    try:
        trainer = Trainer(
            callbacks=[WandbCallback()],
            model=model,
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        results = trainer.evaluate()
        return results
    except ValueError as e:
        print(f"Evaluation error for {model_name}: {e}")
        return None

# Fine-tune the model on the Yelp dataset
def fine_tune_model(model_name, dataset, output_path, max_length):
    print(f"\nFine-tuning model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=5, ignore_mismatched_sizes=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_dataset = preprocess_dataset(dataset, tokenizer, max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8 if torch.cuda.is_available() else 1,
        per_device_eval_batch_size=8 if torch.cuda.is_available() else 1,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_path}/logs",
        report_to=["wandb"],
        run_name=f"{model_name}_fine_tune"
    )

    try:
        trainer = Trainer(
            callbacks=[WandbCallback()],
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(output_path)
        print(f"Fine-tuning completed. Model saved at {output_path}")
    except Exception as e:
        print(f"Fine-tuning error for {model_name}: {e}")

# Main function to evaluate, fine-tune, and re-evaluate models
def main():
    config = load_config()
    dataset = load_dataset("yelp_review_full")

    results = {}
    for model_name, model_config in config["models"].items():
        max_length = model_config["max_length"]

        # Step 1: Baseline Evaluation (Pre-trained Model)
        results[model_name] = {}
        results[model_name]["baseline"] = evaluate_model(model_name, dataset, max_length)

        # Step 2: Fine-Tuning
        output_path = f"fine_tuned_models/{model_name.replace('/', '_')}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        fine_tune_model(model_name, dataset, output_path, max_length)

        # Step 3: Post Fine-Tuning Evaluation
        results[model_name]["fine_tuned"] = evaluate_model(model_name, dataset, max_length, fine_tuned=True)

    # Save results to JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nAll fine-tuning tasks completed successfully! Evaluation results saved to 'evaluation_results.json'.")

if __name__ == "__main__":
    main()
