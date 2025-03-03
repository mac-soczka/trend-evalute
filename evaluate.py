import os
import sys
import subprocess
import datetime
import torch
import yaml
import getpass
from pathlib import Path
from huggingface_hub import login

try:
    import wandb
except ImportError:
    wandb = None

log_file = "eval_log.txt"
error_file = "eval_error.txt"
results_dir = "results"

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
tasks = config.get("tasks", [])
write_log(f"📌 Tasks from config.yml: {tasks}")

HF_CACHE_DIR = config.get("hf_cache_dir", None)
if HF_CACHE_DIR:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
    write_log(f"✅ Using custom HuggingFace cache directory: {HF_CACHE_DIR}")

wandb_config = config.get("wandb", {})
use_wandb = wandb_config.get("enabled", False)
wandb_project = wandb_config.get("project", "lm-eval-project")
wandb_run_name = wandb_config.get("run_name", None)

# ✅ Remove Old Logs
for file_path in [log_file, error_file]:
    if os.path.exists(file_path):
        os.remove(file_path)

os.environ["PYTHONUTF8"] = "1"

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

# ✅ Run Model Evaluation
for model in models:
    base_model_args = f"pretrained={model},max_length=512,truncation='only_first'"

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
    
    write_log(f"🛠️ Running lm-eval for model {model}...")
    try:
        subprocess.run(lm_eval_args, check=True)
        write_log(f"✅ Evaluation for {model} completed.", "SUCCESS")
    except subprocess.CalledProcessError as e:
        write_log(f"❌ lm-eval failed for {model}: {e}", "ERROR")
        sys.exit(1)

# ✅ Zeno Visualization Integration
zeno_config = config.get("zeno", {})
use_zeno = zeno_config.get("enabled", False)

if use_zeno:
    zeno_script_path = Path("lm-evaluation-harness/scripts/zeno_visualize.py")
    if not zeno_script_path.exists():
        write_log(f"❌ zeno_visualize.py not found at expected path: {zeno_script_path}", "ERROR")
        sys.exit(1)

    # Ensure the results directory exists
    if not Path(results_dir).exists():
        write_log(f"❌ Results directory not found: {results_dir}", "ERROR")
        sys.exit(1)

    # Pass the directory path to Zeno
    zeno_args = [
        sys.executable, str(zeno_script_path),
        "--data_path", str(results_dir),  # Pass the directory, not individual files
        "--project_name", "Zeno Visualization - All Results"
    ]

    write_log(f"📊 Running Zeno Visualization for directory: {results_dir}")
    try:
        subprocess.run(zeno_args, check=True)
        write_log("✅ Zeno visualization completed!")
    except subprocess.CalledProcessError as e:
        write_log(f"❌ Zeno visualization failed: {e}", "ERROR")

write_log("🎉 All tasks completed successfully!")