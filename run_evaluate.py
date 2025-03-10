import os
import sys
import subprocess
import datetime
import torch
import yaml
import getpass
from pathlib import Path
from huggingface_hub import login

log_file = "eval_log.txt"
error_file = "eval_error.txt"
results_dir = "results"

def write_log(message, log_type="INFO"):
    """Writes a log message to both console and log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{log_type}] {message}"
    print(formatted_message)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(formatted_message + "\n")

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

# Ensure cache directory is properly set
if HF_CACHE_DIR:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
    if not os.path.exists(HF_CACHE_DIR):
        write_log(f"HF cache directory '{HF_CACHE_DIR}' does not exist. Creating it.", "WARNING")
        os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["PYTHONUTF8"] = "1"

# Authenticate with Hugging Face Hub
hf_token = os.getenv("HF_TOKEN") or getpass.getpass("Enter Hugging Face Token: ")
if not hf_token:
    write_log("Hugging Face Token not provided!", "ERROR")
    sys.exit(1)

login(token=hf_token)

# Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"
write_log(f"Using device: {device}")

# Evaluate Each Model
for model_name, model_props in models.items():
    # Get max length and ensure it's within model constraints
    default_max_length = 512  # Safe fallback if unknown
    model_max_length = model_props.get("max_length", default_max_length)
    model_max_length = min(model_max_length, 2048)  # Avoid overflows in long-context models

    base_model_args = (
        f"pretrained={model_name},max_length={model_max_length},truncation='only_first'"
    )

    if HF_CACHE_DIR:
        base_model_args += f",cache_dir={HF_CACHE_DIR}"

    lm_eval_args = [
        sys.executable, "-m", "lm_eval",
        "--model", "huggingface",
        "--model_args", base_model_args,
        "--tasks", ",".join(tasks),
        "--num_fewshot", "5",
        "--batch_size", "4",
        "--device", device,
        "--output_path", results_dir,
        "--log_samples"
    ]

    write_log(f"Running lm-eval for model {model_name} with max_length={model_max_length}...")
    
    try:
        subprocess.run(lm_eval_args, check=True, stdout=sys.stdout, stderr=sys.stderr)
        write_log(f"✅ Evaluation for {model_name} completed successfully.", "SUCCESS")
    except subprocess.CalledProcessError as e:
        write_log(f"❌ lm-eval failed for {model_name}: {e}", "ERROR")
        sys.exit(1)

# Run Zeno Visualization if Enabled
zeno_config = config.get("zeno", {})
if zeno_config.get("enabled", False):
    zeno_script_path = Path("lm-evaluation-harness/scripts/zeno_visualize.py")
    if not zeno_script_path.exists():
        write_log(f"⚠️ Zeno visualization script not found: {zeno_script_path}", "ERROR")
        sys.exit(1)

    if not Path(results_dir).exists():
        write_log(f"⚠️ Results directory not found: {results_dir}", "ERROR")
        sys.exit(1)

    zeno_args = [
        sys.executable, str(zeno_script_path),
        "--data_path", str(results_dir),
        "--project_name", zeno_config.get("project_name", "Zeno Visualization")
    ]

    write_log(f"📊 Running Zeno Visualization for results in {results_dir}")
    
    try:
        subprocess.run(zeno_args, check=True, stdout=sys.stdout, stderr=sys.stderr)
        write_log("✅ Zeno visualization completed successfully!", "SUCCESS")
    except subprocess.CalledProcessError as e:
        write_log(f"❌ Zeno visualization failed: {e}", "ERROR")

write_log("🎉 All evaluation tasks completed successfully!")
