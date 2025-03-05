import os
import sys
import subprocess
import datetime
import yaml
import getpass
from pathlib import Path

import io
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from huggingface_hub import notebook_login, login
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import Trainer, TrainingArguments
import evaluate

log_file = "eval_log.txt"
error_file = "eval_error.txt"

def write_log(message, log_type="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{log_type}] {message}"
    print(formatted_message)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(formatted_message + "\n")

for file_path in [log_file, error_file]:
    if os.path.exists(file_path):
        os.remove(file_path)

config_file = "config.yml"
if not os.path.exists(config_file):
    write_log(f"Configuration file '{config_file}' not found!", "ERROR")
    sys.exit(1)
with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

hf_token = os.getenv("HF_TOKEN") or getpass.getpass("Enter your Hugging Face Token: ")
if not hf_token:
    write_log("Hugging Face Token not provided!", "ERROR")
    sys.exit(1)
try:
    login(token=hf_token)
    write_log("Successfully logged into Hugging Face.")
except Exception as e:
    write_log(f"Hugging Face login failed: {e}", "ERROR")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
write_log(f"Using device: {device}")

write_log("Loading dataset...")
dataset = load_dataset('pcuenq/oxford-pets')
labels = dataset['train'].unique('label')
write_log(f"Dataset loaded with {len(labels)} labels: {labels}")

def show_samples(ds, rows, cols):
    samples = ds.shuffle().select(np.arange(rows * cols))
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for i in range(rows * cols):
        img_bytes = samples[i]['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        label = samples[i]['label']
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    plt.show()

write_log("Displaying sample images...")
show_samples(dataset['train'], rows=3, cols=5)

write_log("Splitting dataset...")
split_dataset = dataset['train'].train_test_split(test_size=0.2)
eval_dataset = split_dataset['test'].train_test_split(test_size=0.5)
our_dataset = DatasetDict({
    'train': split_dataset['train'],
    'validation': eval_dataset['train'],
    'test': eval_dataset['test']
})
label2id = {c: idx for idx, c in enumerate(labels)}
id2label = {idx: c for idx, c in enumerate(labels)}

write_log("Initializing image processor...")
processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

def transforms(batch):
    batch['image'] = [Image.open(io.BytesIO(x['bytes'])).convert('RGB') for x in batch['image']]
    inputs = processor(batch['image'], return_tensors='pt')
    inputs['labels'] = [label2id[y] for y in batch['label']]
    return inputs

processed_dataset = our_dataset.with_transform(transforms)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

write_log("Loading evaluation metric...")
accuracy = evaluate.load('accuracy')
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    score = accuracy.compute(predictions=predictions, references=labels)
    return score

write_log("Loading pre-trained model...")
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
for name, p in model.named_parameters():
    if not name.startswith('classifier'):
        p.requires_grad = False
num_params = sum([p.numel() for p in model.parameters()])
trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
write_log(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable.")

write_log("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./vit-base-oxford-iiit-pets",
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=5,
    learning_rate=3e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor
)

write_log("Starting training...")
trainer.train()
write_log("Training completed.")

write_log("Evaluating on test dataset...")
eval_results = trainer.evaluate(processed_dataset['test'])
write_log(f"Test evaluation results: {eval_results}")

def show_predictions(rows, cols):
    samples = our_dataset['test'].shuffle().select(np.arange(rows * cols))
    processed_samples = samples.with_transform(transforms)
    predictions = trainer.predict(processed_samples).predictions.argmax(axis=1)
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for i in range(rows * cols):
        img_bytes = samples[i]['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        prediction = predictions[i]
        label = f"label: {samples[i]['label']}\npredicted: {id2label[prediction]}"
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    plt.show()

write_log("Displaying predictions...")
show_predictions(rows=5, cols=5)

write_log("Saving and uploading model...")
kwargs = {
    "finetuned_from": model.config._name_or_path,
    "dataset": "pcuenq/oxford-pets",
    "tasks": "image-classification",
    "tags": ["image-classification"],
}
trainer.save_model()
trainer.push_to_hub("🐕️🐈️", **kwargs)
write_log("Model saved and uploaded successfully.")
write_log("All tasks completed successfully!")
