# Install required packages
#!pip install transformers datasets evaluate

###############################################
# Section 1: Load and Prepare the Dataset
###############################################
from datasets import load_dataset
# Load the Yelp Review Full dataset
dataset = load_dataset("yelp_review_full")
# Display a sample review from the training set
print(dataset["train"][100])

from transformers import AutoTokenizer
# Initialize the tokenizer for a BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    # Tokenize text data with padding and truncation
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization over the entire dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Create small subsets for faster training and evaluation
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

###############################################
# Section 2: Fine-Tuning with PyTorch Trainer
###############################################
from transformers import AutoModelForSequenceClassification
# Load a pretrained BERT model with a classification head for 5 classes
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# Define training hyperparameters and output directory for checkpoints
from transformers import TrainingArguments
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# Set up evaluation metric using the evaluate library
import numpy as np
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    # Compute accuracy from model logits and labels
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Create a Trainer instance for easy fine-tuning
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Start the training process using Trainer
trainer.train()

###############################################
# Section 3: Manual Training Loop in Native PyTorch
###############################################
# Clean up memory by deleting previous model and trainer instances
del model
del trainer
import torch
torch.cuda.empty_cache()

# Reload the small datasets (if needed)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Create DataLoaders for batching data during training and evaluation
from torch.utils.data import DataLoader
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# Reload the model for manual training
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# Set up the optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define a learning rate scheduler based on the number of training steps
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Select the device (GPU if available, otherwise CPU) and move the model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop with progress bar using tqdm
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Move batch to the selected device
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        # Backpropagation
        loss.backward()
        # Optimizer and scheduler steps
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluate the model on the evaluation dataset
import evaluate
metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
# Print the final evaluation accuracy
print("Final evaluation metrics:", metric.compute())

# Save the manually fine-tuned model to a directory
model_save_path = "manual_finetuned_model"
model.save_pretrained(model_save_path)
print(f"Model saved at: {model_save_path}")
