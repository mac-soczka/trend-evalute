import os
import json
import pandas as pd
from pycaret.regression import (
    setup, create_model, tune_model, finalize_model, predict_model,
    save_model, plot_model, pull
)

def load_data(data_folder='data/raw'):
    records = []
    for file in os.listdir(data_folder):
        if file.endswith('.json'):
            with open(os.path.join(data_folder, file)) as f:
                raw = json.load(f)
                if isinstance(raw, list):
                    records.extend(raw)
                elif isinstance(raw, dict) and "features" in raw:
                    records.extend(raw["features"])
    df = pd.DataFrame(records)
    df = df[df["views_per_day"] > 0]  # filter out invalid targets
    df["views_per_day"] = df["views_per_day"].clip(upper=df["views_per_day"].quantile(0.99))
    return df

def train():
    df = load_data()

    setup(
        data=df,
        target="views_per_day",
        session_id=42,
        use_gpu=True,
        log_experiment=True,
        experiment_name="trend-eval-lgbm",
        ignore_features=["id"],
        log_plots=True,
        log_profile=False,
        log_data=True,
    )

    model = create_model('lightgbm', verbose=False)
    tuned = tune_model(model, optimize='R2', choose_better=True)
    final_model = finalize_model(tuned)
    predict_model(final_model)
    save_model(final_model, 'lgbm_model')

    # Save all key charts into a special directory
    charts_dir = os.path.join("charts", "pycaret_charts")
    os.makedirs(charts_dir, exist_ok=True)
    for chart in ['residuals', 'error', 'feature', 'learning', 'vc', 'cooks']:
        try:
            plot_model(final_model, plot=chart, save=True)
            os.replace(f'{chart}.png', os.path.join(charts_dir, f'{chart}.png'))
        except Exception as e:
            print(f"Failed to generate {chart}: {e}")

if __name__ == "__main__":
    train()
