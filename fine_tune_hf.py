# -*- coding: utf-8 -*-
"""Refactored fine-tuning script for a pretrained model."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_scheduler,
)
import evaluate


class FineTuningPipeline:
    def __init__(self, model_name="bert-base-cased", dataset_name="yelp_review_full", num_labels=5):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_labels = num_labels
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.model.to(self.device)

    def prepare_data(self, sample_size=1000):
        """Load and preprocess the dataset."""
        dataset = load_dataset(self.dataset_name)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(sample_size))
        small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(sample_size))

        return small_train_dataset, small_eval_dataset

    def create_dataloaders(self, train_dataset, eval_dataset, batch_size=8):
        """Create DataLoader objects for training and evaluation."""
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        return train_dataloader, eval_dataloader

    def compute_metrics(self, eval_pred):
        """Compute accuracy for evaluation."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=predictions, references=labels)

    def train_with_trainer(self, train_dataset, eval_dataset, output_dir="test_trainer", epochs=3):
        """Train the model using Hugging Face Trainer."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=epochs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    def train_with_pytorch(self, train_dataloader, eval_dataloader, epochs=3, lr=5e-5):
        """Train the model using native PyTorch."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))
        self.model.train()

        for epoch in range(epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

    def evaluate(self, eval_dataloader):
        """Evaluate the model."""
        metric = evaluate.load("accuracy")
        self.model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        return metric.compute()


def main():
    # Initialize the pipeline
    pipeline = FineTuningPipeline()

    # Prepare data
    train_dataset, eval_dataset = pipeline.prepare_data(sample_size=1000)

    # Option 1: Train using Hugging Face Trainer
    pipeline.train_with_trainer(train_dataset, eval_dataset)

    # Option 2: Train using native PyTorch
    train_dataloader, eval_dataloader = pipeline.create_dataloaders(train_dataset, eval_dataset)
    pipeline.train_with_pytorch(train_dataloader, eval_dataloader)

    # Evaluate the model
    accuracy = pipeline.evaluate(eval_dataloader)
    print(f"Model accuracy: {accuracy}")


if __name__ == "__main__":
    main()