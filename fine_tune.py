import os
import sys
import torch
import datetime
import yaml
import getpass
from pathlib import Path
from transformers import (
    Trainer, TrainingArguments, AutoModelForSequenceClassification,
    AutoTokenizer, DataCollatorWithPadding
)
from datasets import load_dataset
from huggingface_hub import login

try:
    import wandb
except ImportError:
    wandb = None

# ✅ Logging setup
log_file = "fine_tune_log.txt"
data_dir = "data"
output_dir = "fine_tuned_models"

def write_log(message, log_type="INFO"):
    """Write log messages to both console and file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{log_type}] {message}"
    print(formatted_message)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(formatted_message + "\n")

# ✅ Load Configuration
config_file = "config.yml"
if not os.path.exists(config_file):
    write_log(f"❌ Configuration file '{config_file}' not found!", "ERROR")
    sys.exit(1)

with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

models = config.get("models", [])
hf_cache_dir = config.get("hf_cache_dir", None)

if hf_cache_dir:
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
    write_log(f"✅ Using custom HuggingFace cache directory: {hf_cache_dir}")

# ✅ Hugging Face Authentication
hf_token = os.getenv("HF_TOKEN") or getpass.getpass("🔑 Enter your Hugging Face Token: ")
if not hf_token:
    write_log("❌ Hugging Face Token not provided!", "ERROR")
    sys.exit(1)

try:
    login(token=hf_token)
    write_log("✅ Successfully logged into Hugging Face.")
except Exception as e:
    write_log(f"❌ Hugging Face login failed: {e}", "ERROR")
    sys.exit(1)

# ✅ CUDA Check
device = "cuda" if torch.cuda.is_available() else "cpu"
write_log(f"🚀 Using device: {device}")

# ✅ Validate dataset files
train_path = os.path.join(data_dir, "train.json")
valid_path = os.path.join(data_dir, "valid.json")

if not (os.path.exists(train_path) and os.path.exists(valid_path)):
    write_log(f"❌ Dataset files not found! Ensure both '{train_path}' and '{valid_path}' exist.", "ERROR")
    sys.exit(1)

# ✅ Load dataset
try:
    dataset = load_dataset("json", data_files={"train": train_path, "validation": valid_path})
    write_log("✅ Successfully loaded dataset.")
except Exception as e:
    write_log(f"❌ Failed to load dataset: {e}", "ERROR")
    sys.exit(1)

# ✅ Fine-tuning each model
for model_name in models:
    write_log(f"🛠️ Fine-tuning model: {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
    except Exception as e:
        write_log(f"❌ Failed to load model/tokenizer for {model_name}: {e}", "ERROR")
        continue

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    try:
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
    except Exception as e:
        write_log(f"❌ Tokenization failed for {model_name}: {e}", "ERROR")
        continue

    # ✅ Set up Trainer
    output_path = f"{output_dir}/{model_name.replace('/', '_')}"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_path}/logs",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        save_total_limit=2,  # Prevent excessive checkpoints
        logging_strategy="steps",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # ✅ Start training
    try:
        trainer.train()
        write_log(f"✅ Fine-tuning completed for {model_name}. Model saved at {training_args.output_dir}")

        # ✅ Evaluate
        eval_results = trainer.evaluate()
        write_log(f"✅ Evaluation results for {model_name}: {eval_results}")

    except Exception as e:
        write_log(f"❌ Training failed for {model_name}: {e}", "ERROR")
        continue

write_log("🎉 All fine-tuning tasks completed successfully!")
