import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === 1. Load data ===
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
    df = df[df["views_per_day"] > 0]
    df["views_per_day"] = df["views_per_day"].clip(upper=df["views_per_day"].quantile(0.99))
    return df

# === 2. Preprocess + Train ===
def train_model(df):
    y = df["views_per_day"]
    X = df.drop(columns=["views_per_day", "id"], errors="ignore")

    categorical = ["genre", "budget_estimate", "dominant_pitch_class", "upload_day_of_week"]
    numeric = X.select_dtypes(include=["number"]).columns.difference(categorical).tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric),
        ("cat", categorical_transformer, categorical)
    ])

    model = LGBMRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    param_grid = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__num_leaves": [20, 31, 40],
        "regressor__max_depth": [-1, 5, 10],
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid,
        scoring="r2", cv=5, n_iter=20, verbose=1, n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    logging.info(f"Best parameters: {search.best_params_}")
    return best_model, X_test, y_test

# === 3. Evaluation and Charts ===
def evaluate_and_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)

    os.makedirs("charts/sklearn", exist_ok=True)

    results = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
        "residual": y_test - y_pred
    })

    results.to_csv("charts/sklearn/predictions.csv", index=False)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    pd.DataFrame([metrics]).to_csv("charts/sklearn/metrics.csv", index=False)
    logging.info(f"Metrics: {metrics}")

    # Residual plot
    plt.figure(figsize=(8, 5))
    sns.histplot(results["residual"], kde=True)
    plt.title("Residuals Distribution")
    plt.savefig("charts/sklearn/residuals_distribution.png")
    plt.close()

    # Actual vs Predicted
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=results["actual"], y=results["predicted"])
    plt.plot([results["actual"].min(), results["actual"].max()],
             [results["actual"].min(), results["actual"].max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.savefig("charts/sklearn/actual_vs_predicted.png")
    plt.close()

# === 4. Entry point ===
if __name__ == "__main__":
    df = load_data()
    model, X_test, y_test = train_model(df)
    evaluate_and_plot(model, X_test, y_test)
