import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load JSON Data ===
def load_dataset(data_folder='data/raw', schema_path='features_schema.json'):
    all_data = []
    with open(schema_path, 'r') as f:
        features = [feat['name'] for feat in json.load(f)]

    for file in os.listdir(data_folder):
        if file.endswith(".json"):
            with open(os.path.join(data_folder, file), 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and "features" in data:
                    data = data["features"]
                elif not isinstance(data, list):
                    continue
                for item in data:
                    row = {k: item.get(k, None) for k in features}
                    all_data.append(row)
    
    return pd.DataFrame(all_data)

df = load_dataset()

# === Clean Up ===
df = df.dropna(subset=["views_per_day"])
df["views_per_day"] = df["views_per_day"].clip(upper=df["views_per_day"].quantile(0.99))

# === Setup Plotting ===
sns.set(style="whitegrid")
output_dir = os.path.join("charts", "EDA")
os.makedirs(output_dir, exist_ok=True)

# === 1. Distribution of Target Variable ===
plt.figure(figsize=(8, 5))
sns.histplot(df["views_per_day"], bins=50, kde=True)
plt.title("Distribution of Views per Day")
plt.xlabel("Views per Day")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "views_per_day_distribution.png"))

# === 2. Correlation with Views per Day ===
plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include='number').drop(columns=["views_per_day"], errors="ignore")
correlation = numeric_df.corrwith(df["views_per_day"]).sort_values(ascending=False)
sns.barplot(x=correlation.values, y=correlation.index)
plt.title("Correlation with Views per Day")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_with_views.png"))

# === 3. Sentiment vs Views ===
if "avg_sentiment_score" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="avg_sentiment_score", y="views_per_day", data=df)
    plt.title("Average Sentiment Score vs Views per Day")
    plt.savefig(os.path.join(output_dir, "sentiment_vs_views.png"))

# === 4. Engagement Rate vs Views ===
if "engagement_rate" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="engagement_rate", y="views_per_day", data=df)
    plt.title("Engagement Rate vs Views per Day")
    plt.savefig(os.path.join(output_dir, "engagement_vs_views.png"))

# === 5. Genre Breakdown ===
if "genre" in df.columns:
    top_genres = df["genre"].value_counts().nlargest(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_genres.index, y=top_genres.values)
    plt.title("Top 10 Genres by Video Count")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Videos")
    plt.savefig(os.path.join(output_dir, "top_genres.png"))

# === 6. Budget Estimate Impact on Views ===
if "budget_estimate" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="budget_estimate", y="views_per_day", data=df)
    plt.title("Views per Day by Budget Estimate")
    plt.savefig(os.path.join(output_dir, "budget_vs_views.png"))

# === 7. Upload Day of Week vs Views ===
if "upload_day_of_week" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="upload_day_of_week", y="views_per_day", data=df)
    plt.title("Views per Day by Upload Day")
    plt.savefig(os.path.join(output_dir, "upload_day_vs_views.png"))

# === 8. Clickbait Impact ===
if "clickbait_thumbnail" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="clickbait_thumbnail", y="views_per_day", data=df)
    plt.title("Clickbait Thumbnail vs Views per Day")
    plt.savefig(os.path.join(output_dir, "clickbait_vs_views.png"))

# === 9. Heatmap of Full Correlation Matrix ===
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include="number").corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))

print(f"EDA complete. All charts saved to: {output_dir}")
