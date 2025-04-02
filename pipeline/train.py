import os
import json
import pandas as pd
import logging
from pycaret.regression import setup, compare_models, finalize_model, predict_model, save_model

# Configure basic logging to a file
log_file_name = "pycaret_training.log"
logging.basicConfig(
    filename=log_file_name,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
logger.info("Logging started for PyCaret training pipeline")

def load_data(data_folder='data/raw'):
    data = []
    for file_name in os.listdir(data_folder):
        print(file_name)
        if file_name.endswith(".json"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                elif isinstance(file_data, dict) and "features" in file_data:
                    data.extend(file_data["features"])
                else:
                    print(f"Skipping {file_name}: unexpected structure")
    return pd.DataFrame(data)

def setup_pycaret(data: pd.DataFrame, target: str):
    setup(
        data=data,
        target=target,
        session_id=123,
        use_gpu=True,
        log_experiment=False  # disable MLflow logging to avoid crash
    )

def train_model():
    data = load_data()
    target = 'views_per_day'
    setup_pycaret(data, target)
    best_model = compare_models()
    final_model = finalize_model(best_model)
    predictions = predict_model(final_model)
    save_model(final_model, 'final_model')
    return predictions

if __name__ == "__main__":
    predictions = train_model()
    print(predictions.head())
