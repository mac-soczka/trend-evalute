import os
import sys
import subprocess
import datetime
from pathlib import Path

log_file = "zeno_upload_log.txt"
error_file = "zeno_upload_error.txt"
results_dir = "results"

def write_log(message, log_type="INFO"):
    """Write log messages to both console and file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{log_type}] {message}"
    print(formatted_message)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(formatted_message + "\n")

# ✅ Ensure ZENO_API_KEY is set
zeno_api_key = os.getenv("ZENO_API_KEY")
if not zeno_api_key:
    write_log("❌ ZENO_API_KEY not set! Please set it in the environment.", "ERROR")
    sys.exit(1)

# ✅ Find the Zeno visualization script
zeno_script_path = os.path.join("lm-evaluation-harness", "scripts", "zeno_visualize.py")
if not os.path.exists(zeno_script_path):
    write_log(f"❌ zeno_visualize.py not found at expected path: {zeno_script_path}", "ERROR")
    sys.exit(1)

# ✅ Ensure the results directory exists
if not os.path.exists(results_dir) or not os.listdir(results_dir):
    write_log(f"❌ No evaluation results found in '{results_dir}'.", "ERROR")
    sys.exit(1)

# ✅ Get all LLM subdirectories inside `results/`
llm_dirs = [f for f in os.scandir(results_dir) if f.is_dir()]
if not llm_dirs:
    write_log(f"❌ No LLM result directories found in '{results_dir}'.", "ERROR")
    sys.exit(1)

write_log(f"✅ Found LLM results: {[d.name for d in llm_dirs]}")

# ✅ Function to find the latest result file in each LLM directory
def get_latest_result_file(llm_dir):
    """Find the latest result JSON file inside an LLM results folder."""
    result_files = list(Path(llm_dir).glob("*.json"))
    
    if not result_files:
        return None  # Return None if no result files are found
    
    return max(result_files, key=lambda f: f.stat().st_mtime)  # Get the newest file

# ✅ Collect all valid result files
valid_models = {}
for llm_dir in llm_dirs:
    latest_result = get_latest_result_file(llm_dir)
    
    if latest_result:
        valid_models[llm_dir.name] = latest_result.name
        write_log(f"📊 Found latest result file for {llm_dir.name}: {latest_result.name}")
    else:
        write_log(f"⚠️ No valid results found in {llm_dir.name}. Skipping...", "WARNING")

# ✅ Ensure we have at least one valid result
if not valid_models:
    write_log(f"❌ No valid JSON result files found in '{results_dir}'.", "ERROR")
    sys.exit(1)

# ✅ Run Zeno Visualization **Per Model** to avoid dataset inconsistencies
for model_name, result_file in valid_models.items():
    model_results_path = os.path.join(results_dir, model_name)

    zeno_args = [
        sys.executable, zeno_script_path,
        "--data_path", model_results_path,
        "--project_name", f"Zeno Visualization - {model_name}"
    ]

    write_log(f"📡 Uploading results to Zeno for model: {model_name} ...")
    try:
        subprocess.run(zeno_args, check=True)
        write_log(f"✅ Successfully uploaded {model_name} results to Zeno.")
    except subprocess.CalledProcessError as e:
        write_log(f"❌ Zeno visualization failed for {model_name}: {e}", "ERROR")

write_log("🎉 Zeno process completed!")
