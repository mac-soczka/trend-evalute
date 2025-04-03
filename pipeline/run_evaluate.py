import os
import sys
import subprocess
import datetime
import torch
import yaml
import getpass
from pathlib import Path
from huggingface_hub import login
from transformers import AutoTokenizer

log_file = "eval_log.txt"
results_dir = "results"

def write_log(message, log_type="INFO"):
    """Writes a log message to both console and log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{log_type}] {message}"
    print(formatted_message)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(formatted_message + "\n")

def get_effective_max_length(model_name, config_max, hf_cache):
    """Get safe max length considering both config and model limitations"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=hf_cache,
            trust_remote_code=True
        )
        model_max = tokenizer.model_max_length
        if model_max > 1000000:
            model_max = 4096 if "llama" in model_name.lower() else 512
        return min(config_max, model_max)
    except Exception as e:
        write_log(f"⚠️ Failed to get max length for {model_name}: {e}, using safe default", "WARNING")
        return min(config_max, 512)

# Load Configuration
config_file = "config.yml"
if not os.path.exists(config_file):
    write_log(f"Configuration file '{config_file}' not found!", "ERROR")
    sys.exit(1)

with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

models = config.get("models", {})
tasks = config.get("tasks", [])
HF_CACHE_DIR = config.get("hf_cache_dir", None)

# Configure environment
os.environ["PYTHONUTF8"] = "1"
if HF_CACHE_DIR:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")

# Authentication
hf_token = os.getenv("HF_TOKEN") or getpass.getpass("Enter Hugging Face Token: ")
if not hf_token:
    write_log("Hugging Face Token not provided!", "ERROR")
    sys.exit(1)
login(token=hf_token)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4 if device == "cuda" else 1
write_log(f"Using device: {device} with batch size: {batch_size}")

# Evaluation pipeline
for model_name, model_props in models.items():
    config_max = model_props.get("max_length", 512)
    effective_max = get_effective_max_length(model_name, config_max, HF_CACHE_DIR)
    
    model_args = [
        f"pretrained={model_name}",
        f"max_length={effective_max}",
        "truncation=only_first",
        "trust_remote_code=True"
    ]
    
    if HF_CACHE_DIR:
        model_args.append(f"cache_dir={HF_CACHE_DIR}")

    lm_eval_args = [
        sys.executable, "-m", "lm_eval",
        "--model", "huggingface",
        "--model_args", ",".join(model_args),
        "--tasks", ",".join(tasks),
        "--num_fewshot", "5",
        "--device", device,
        "--output_path", results_dir,
        "--log_samples"
    ]

    write_log(f"🚀 Starting evaluation for {model_name} (max_length={effective_max})...")
    
    try:
        subprocess.run(
            lm_eval_args,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=3600  # 1 hour timeout per model
        )
        write_log(f"✅ Successfully evaluated {model_name}", "SUCCESS")
    except subprocess.CalledProcessError as e:
        write_log(f"❌ Evaluation failed for {model_name}: {e}", "ERROR")
    except subprocess.TimeoutExpired:
        write_log(f"⏰ Timeout exceeded for {model_name}", "WARNING")

# Zeno visualization
zeno_config = config.get("zeno", {})
if zeno_config.get("enabled", False):
    zeno_script_path = Path("lm-evaluation-harness/scripts/zeno_visualize.py")
    if zeno_script_path.exists():
        zeno_args = [
            sys.executable, str(zeno_script_path),
            "--data_path", results_dir,
            "--project_name", zeno_config.get("project_name", "Zeno Visualization")
        ]
        write_log("📊 Generating Zeno visualization...")
        subprocess.run(zeno_args, check=True)
    else:
        write_log("⚠️ Zeno visualization script not found", "WARNING")

write_log("🎉 All tasks completed!", "SUCCESS")
