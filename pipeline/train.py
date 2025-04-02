import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import (
    setup,
    create_model,
    tune_model,
    finalize_model,
    predict_model,
    save_model,
    plot_model,
    pull
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(data_folder='data/raw'):
    records = []
    for file in os.listdir(data_folder):
        if file.endswith('.json'):
            try:
                with open(os.path.join(data_folder, file)) as f:
                    raw = json.load(f)
                    if isinstance(raw, list):
                        records.extend(raw)
                    elif isinstance(raw, dict) and "features" in raw:
                        records.extend(raw["features"])
            except Exception as e:
                logging.warning(f"Skipping {file}: {e}")
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Data is empty or malformed.")
    if 'views_per_day' not in df.columns:
        raise KeyError("Target column 'views_per_day' missing.")

    # Handle infinite or negative values, clip if needed
    df = df[df["views_per_day"] > 0]  # Yeo-Johnson needs positive data if applied
    df["views_per_day"] = df["views_per_day"].clip(upper=df["views_per_day"].quantile(0.99))  # optional

    return df

def setup_experiment(df):
    return setup(
        data=df,
        target="views_per_day",
        session_id=42,
        train_size=0.85,
        fold_strategy="kfold",
        fold=5,
        use_gpu=True,
        normalize=True,
        transformation=True,
        transform_target=False,  # DISABLED to prevent crash
        ignore_features=["id"],
        remove_multicollinearity=True,
        multicollinearity_threshold=0.85,
        feature_selection=True,
        low_variance_threshold=0.01,
        categorical_features=[
            "genre", "budget_estimate", "dominant_pitch_class", "upload_day_of_week"
        ],
        verbose=False,
        log_experiment=True,
        experiment_name="trend-eval-lgbm",
        log_plots=True
    )

def tune_lgbm():
    model = create_model('lightgbm', verbose=False)
    tuned = tune_model(
        model,
        optimize="R2",
        choose_better=True,
        search_library="scikit-optimize",
        n_iter=40
    )
    return finalize_model(tuned)

def generate_advanced_charts(model, predictions):
    os.makedirs("charts", exist_ok=True)
    for plot in ['residuals', 'error', 'feature', 'learning', 'vc', 'cooks']:
        try:
            plot_model(model, plot=plot, save=True)
        except Exception as e:
            logging.warning(f"Failed to plot {plot}: {e}")
    for f in os.listdir():
        if f.endswith(".png"):
            os.replace(f, os.path.join("charts", f))

    try:
        residuals = predictions["Label"] - predictions["prediction_label"]
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title("Residuals Distribution")
        plt.savefig("charts/custom_residuals_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=predictions["Label"], y=predictions["prediction_label"])
        plt.plot([predictions["Label"].min(), predictions["Label"].max()],
                 [predictions["Label"].min(), predictions["Label"].max()], 'r--')
        plt.title("Actual vs Predicted")
        plt.savefig("charts/custom_actual_vs_predicted.png")
        plt.close()

    except Exception as e:
        logging.warning(f"Custom plots failed: {e}")

def train():
    df = load_data()
    setup_experiment(df)
    model = tune_lgbm()
    predictions = predict_model(model)
    save_model(model, 'world_class_lightgbm')
    generate_advanced_charts(model, predictions)
    metrics = pull()
    predictions.to_csv("charts/predictions.csv", index=False)
    metrics.to_csv("charts/metrics.csv")
    logging.info("Training complete. Metrics:\n%s", metrics)

if __name__ == "__main__":
    train()
