import os
import torch
import yaml
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def load_config(config_file="config.yml"):
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def evaluate_model(model_name, dataset_files, max_length, fine_tuned=False):
    print(f"\nEvaluating {'fine-tuned' if fine_tuned else 'pretrained'} model: {model_name}...")
    
    model_path = f"fine_tuned_models/{model_name.replace('/', '_')}" if fine_tuned else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("json", data_files=dataset_files)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    trainer = Trainer(
        model=model,
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer
    )
    results = trainer.evaluate()
    return results

def fine_tune_model(model_name, dataset_files, output_path, max_length):
    print(f"\nFine-tuning model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = load_dataset("json", data_files=dataset_files)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_path}/logs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model(output_path)
    print(f"✅ Fine-tuning completed. Model saved at {output_path}")

def main():
    config = load_config()
    dataset_files = {"train": "data/train.json", "validation": "data/valid.json"}
    
    results = {}
    for model_name, model_config in config["models"].items():
        max_length = model_config["max_length"]
        
        # Step 1: Baseline Evaluation
        results[model_name] = {}
        results[model_name]["baseline"] = evaluate_model(model_name, dataset_files, max_length)
        
        # Step 2: Fine-Tuning
        output_path = f"fine_tuned_models/{model_name.replace('/', '_')}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        fine_tune_model(model_name, dataset_files, output_path, max_length)
        
        # Step 3: Post Fine-Tuning Evaluation
        results[model_name]["fine_tuned"] = evaluate_model(model_name, dataset_files, max_length, fine_tuned=True)
    
    # Save results to JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print("\n🎉 All fine-tuning tasks completed successfully! Evaluation results saved to 'evaluation_results.json'.")

if __name__ == "__main__":
    main()
