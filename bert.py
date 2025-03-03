import os
import sys
import subprocess
import datetime
import torch
from huggingface_hub import login
import getpass

# ===========================
# 1️⃣ Define Log Function
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
# 2️⃣ Cleanup Previous Logs
# ===========================
for file in [log_file, error_file]:
    if os.path.exists(file):
        os.remove(file)

# ===========================
# 3️⃣ Ensure Python Environment
# ===========================
os.environ["PYTHONUTF8"] = "1"

# ===========================
# 4️⃣ Retrieve Hugging Face Token
# ===========================
hf_token = os.getenv("HF_TOKEN") or getpass.getpass("🔑 Enter your Hugging Face Token: ")
if not hf_token:
    write_log("❌ Hugging Face Token not provided!", "ERROR")
    sys.exit(1)
write_log("🔑 Hugging Face Token retrieved successfully.")

# ===========================
# 5️⃣ Log in to Hugging Face
# ===========================
login(token=hf_token)
write_log("✅ Successfully logged into Hugging Face.")

# ===========================
# 6️⃣ Check CUDA Availability
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    write_log("⚠️ CUDA not available. Running on CPU.", "WARNING")
else:
    write_log("✅ CUDA is available. Running on GPU.")

# ===========================
# 7️⃣ Define lmms_eval Parameters
# ===========================
lmms_eval_args = [
    sys.executable, "-m", "lm_eval",
    "--model", "huggingface",
    "--model_args", "pretrained=nlptown/bert-base-multilingual-uncased-sentiment,max_length=512,truncation=True",
    "--tasks", "hellaswag,mmlu",
    "--num_fewshot", "5",
    "--batch_size", "8",
    "--device", device,
    "--output_path", results_dir
]

write_log("🚀 Starting evaluation with lm-eval...")
write_log(f"Using device: {device}")
write_log(f"Running lm-eval with parameters: {' '.join(lmms_eval_args)}")

# ===========================
# 8️⃣ Run lm-eval and Capture Output
# ===========================
try:
    with open(log_file, "w", encoding="utf-8") as log, open(error_file, "w", encoding="utf-8") as err:
        subprocess.run(lmms_eval_args, stdout=log, stderr=err, text=True, check=True)

    if os.path.exists(results_dir) and os.listdir(results_dir):
        write_log(f"✅ Evaluation completed successfully. Results saved to {results_dir}.", "SUCCESS")
    else:
        write_log(f"❌ Evaluation failed. Check '{log_file}' and '{error_file}' for details.", "ERROR")

except subprocess.CalledProcessError as e:
    write_log(f"❌ Error running lm-eval: {e}", "ERROR")
    sys.exit(1)

# ===========================
# 9️⃣ Display Logs Summary
# ===========================
write_log(f"📜 Logs saved as '{log_file}'. View logs with: open('{log_file}')")
