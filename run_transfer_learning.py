# Section 1: Imports and Setup
import io
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import Trainer, TrainingArguments
import run_evaluate


print("OK")
exit()

notebook_login()

# Section 2: Load Dataset and Labels
dataset = load_dataset('pcuenq/oxford-pets')
labels = dataset['train'].unique('label')
print(len(labels), labels)

# Section 3: Display Sample Images
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

show_samples(dataset['train'], rows=3, cols=5)

# Section 4: Split Dataset and Map Labels
split_dataset = dataset['train'].train_test_split(test_size=0.2)
eval_dataset = split_dataset['test'].train_test_split(test_size=0.5)
our_dataset = DatasetDict({
    'train': split_dataset['train'],
    'validation': eval_dataset['train'],
    'test': eval_dataset['test']
})
label2id = {c: idx for idx, c in enumerate(labels)}
id2label = {idx: c for idx, c in enumerate(labels)}

# Section 5: Initialize Image Processor
processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Section 6: Preprocessing Function
def transforms(batch):
    batch['image'] = [Image.open(io.BytesIO(x['bytes'])).convert('RGB') for x in batch['image']]
    inputs = processor(batch['image'], return_tensors='pt')
    inputs['labels'] = [label2id[y] for y in batch['label']]
    return inputs

processed_dataset = our_dataset.with_transform(transforms)

# Section 7: Data Collation Function
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# Section 8: Metrics Calculation
accuracy = run_evaluate.load('accuracy')
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    score = accuracy.compute(predictions=predictions, references=labels)
    return score

# Section 9: Load and Configure Model
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
print(f"{num_params = :,} | {trainable_params = :,}")

# Section 10: Training Setup
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
trainer.train()

# Section 11: Evaluation on Test Dataset
trainer.evaluate(processed_dataset['test'])

# Section 12: Predictions and Display
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

show_predictions(rows=5, cols=5)

# Section 13: Save and Upload Model
kwargs = {
    "finetuned_from": model.config._name_or_path,
    "dataset": 'pcuenq/oxford-pets',
    "tasks": "image-classification",
    "tags": ['image-classification'],
}
trainer.save_model()
trainer.push_to_hub('🐕️🐈️', **kwargs)
