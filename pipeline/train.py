import os
import json
import pandas as pd
from pycaret.regression import (
    setup,
    create_model,
    finalize_model,
    predict_model,
    save_model,
    plot_model
)

def load_data(data_folder='data/raw'):
    data = []
    for file_name in os.listdir(data_folder):
        print(f"Loading {file_name}")
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
        log_experiment=False,
        html=True
    )

def generate_visuals(model):
    os.makedirs("charts", exist_ok=True)

    plot_model(model, plot='residuals', save=True)
    plot_model(model, plot='error', save=True)
    plot_model(model, plot='feature', save=True)
    plot_model(model, plot='learning', save=True)
    plot_model(model, plot='rfe', save=True)
    plot_model(model, plot='cooks', save=True)
    plot_model(model, plot='manifold', save=True)
    plot_model(model, plot='vc', save=True)

    for file in os.listdir():
        if file.endswith(".png"):
            os.replace(file, os.path.join("charts", file))

def train_model():
    data = load_data()
    target = 'views_per_day'
    setup_pycaret(data, target)

    model = create_model(
        'lightgbm',
        verbose=False,
        gpu_use_dp=True,
        device='gpu'
    )

    final_model = finalize_model(model)
    predictions = predict_model(final_model)

    save_model(final_model, 'final_model')
    generate_visuals(final_model)

    return predictions

if __name__ == "__main__":
    predictions = train_model()
    print(predictions.head())
