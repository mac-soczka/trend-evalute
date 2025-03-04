import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def setup_environment(model_name, cache_dir="hf_cache"):
    """Set up the environment for evaluation."""
    logger.info("Setting up environment...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"🚀 Using device: {device}")

    # Load model and tokenizer
    logger.info(f"🛠️ Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    # Set custom cache directory
    logger.info(f"✅ Using custom HuggingFace cache directory: {cache_dir}")
    return model, tokenizer, device

def run_evaluation(model, tokenizer, device, tasks):
    """Run evaluation on the specified tasks."""
    logger.info("🛠️ Running lm-eval...")
    logger.info(f"📌 Tasks: {tasks}")

    for task in tasks:
        logger.info(f"🔍 Evaluating task: {task}")
        # Add your evaluation logic here
        # Example: result = evaluate_task(model, tokenizer, device, task)
        logger.info(f"✅ Completed task: {task}")

    logger.info("🎉 Evaluation complete.")

def main():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tasks = ["hellaswag", "mmlu", "arc_challenge", "lambada", "winogrande", "piqa"]
    cache_dir = "hf_cache"

    # Set up environment
    model, tokenizer, device = setup_environment(model_name, cache_dir)

    # Run evaluation
    run_evaluation(model, tokenizer, device, tasks)

if __name__ == "__main__":
    main()