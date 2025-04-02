import os
import pandas as pd
from pycaret.regression import *  # Use `classification` if it's a classification task
import json

# Load raw data from the folder
def load_data(data_folder='data/raw'):
    data = []
    for file_name in os.listdir(data_folder):
        print(file_name)
        if file_name.endswith(".json"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r') as f:
                data.append(json.load(f))
    return pd.DataFrame(data)

# Setup PyCaret
def setup_pycaret(data: pd.DataFrame, target: str):
    setup(data=data, target=target, session_id=123, use_gpu=False)

# Train model
def train_model():
    # Load the data
    data = load_data()
    
    # Set the target column (e.g., views, or any other column for prediction)
    target = 'views_per_day'  # Example target, replace with the actual target column
    
    # Setup PyCaret
    setup_pycaret(data, target)
    
    # Compare models (to select the best model based on performance)
    best_model = compare_models()
    
    # Finalize model
    final_model = finalize_model(best_model)
    
    # Predict with the model
    predictions = predict_model(final_model)
    
    # Save the model
    save_model(final_model, 'final_model')

    return predictions

if __name__ == "__main__":
    predictions = train_model()
    print(predictions.head())
