import os
import sys
import shutil
import subprocess
import datetime
import torch
from huggingface_hub import login
from datasets import load_dataset
import getpass

# ===========================
# 1️⃣ Install Dependencies
# ===========================
def install_packages(packages):
    """Install missing packages dynamically."""
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} is already installed.")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

# List of required packages
required_packages = ["torch", "lmms_eval", "datasets", "huggingface_hub"]
install_packages(required_packages)

# ===========================
# 2️⃣ Retrieve Hugging Face Token
# ===========================
hf_token = os.getenv("HF_TOKEN") or getpass.getpass("🔑 Enter your Hugging Face Token: ")

if not hf_token:
    print("❌ Hugging Face Token not provided!")
    sys.exit(1)

print("🔑 Hugging Face Token retrieved successfully.")

# ===========================
# 3️⃣ Log in to Hugging Face
# ===========================
login(token=hf_token)
print("✅ Successfully logged into Hugging Face.")

# ===========================
# 4️⃣ Set Dataset Cache Directory
# ===========================
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

if os.name == "nt":  # Windows
    cache_dir = os.path.join(os.getenv("LOCALAPPDATA"), "huggingface", "datasets")

os.makedirs(cache_dir, exist_ok=True)  # Ensure the cache directory exists
print(f"📂 Using dataset cache directory: {cache_dir}")

# ===========================
# 5️⃣ Load VQAv2 Dataset with Increased Timeout
# ===========================
def load_vqav2_dataset(retries=3):
    """Load the VQAv2 dataset with increased timeout and local caching."""
    for attempt in range(retries):
        try:
            print("📥 Downloading VQAv2 dataset...")
            dataset = load_dataset("lmms-lab/vqav2", token=hf_token, cache_dir=cache_dir)
            print("✅ VQAv2 dataset loaded successfully!")
            return dataset

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print("🔄 Retrying in 10 seconds...")
                import time
                time.sleep(10)  # Wait before retrying
            else:
                print("❌ Failed to load VQAv2 dataset after multiple attempts.")
                print("🔗 Try manually downloading the dataset: https://huggingface.co/datasets/lmms-lab/vqav2")
                sys.exit(1)

# Load dataset with retry mechanism
dataset = load_vqav2_dataset()

# ===========================
# 6️⃣ Define Log Function
# ===========================
log_file = "eval_log.txt"
error_file = "eval_error.txt"
results_dir = "results"

def write_log(message, log_type="INFO"):
    """Write logs to console and a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{log_type}] {message}"
    print(formatted_message)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(formatted_message + "\n")

# ===========================
# 7️⃣ Check CUDA Availability
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    write_log("⚠️ CUDA not available. Running on CPU.", "WARNING")
else:
    write_log("✅ CUDA is available. Running on GPU.", "INFO")

# ===========================
# 8️⃣ Run lmms_eval Using `sys.executable`
# ===========================
lmms_eval_args = [
    sys.executable, "-m", "lmms_eval",
    "--model", "llava",
    "--model_args", f"pretrained=liuhaotian/llava-v1.5-7b",
    "--tasks", "vqav2",
    "--num_fewshot", "5",
    "--batch_size", "4",
    "--device", device,
    "--output_path", results_dir
]

write_log("🚀 Starting evaluation with lmms_eval...")
write_log(f"Using device: {device}")
write_log(f"Running lmms_eval with parameters: {' '.join(lmms_eval_args)}")

try:
    with open(log_file, "w", encoding="utf-8") as log, open(error_file, "w", encoding="utf-8") as err:
        subprocess.run(lmms_eval_args, stdout=log, stderr=err, text=True, check=True)

    if os.path.exists(results_dir):
        write_log(f"✅ Evaluation completed successfully. Results saved to {results_dir}.", "SUCCESS")
        print(f"✅ Evaluation completed successfully. Results saved to {results_dir}.")
    else:
        write_log(f"❌ Evaluation failed. Check '{log_file}' and '{error_file}' for details.", "ERROR")
        print(f"❌ Evaluation failed. Check '{log_file}' and '{error_file}' for details.", file=sys.stderr)

except subprocess.CalledProcessError as e:
    write_log(f"❌ Error running lmms_eval: {e}", "ERROR")
    print(f"❌ Error running lmms_eval: {e}", file=sys.stderr)
    sys.exit(1)

# ===========================
# 9️⃣ Display Logs Summary
# ===========================
write_log(f"📜 Logs saved as '{log_file}'. View logs with: open('{log_file}')")
