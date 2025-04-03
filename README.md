# 🎬 Trend-Eval: Predicting Content Success Using Hybrid ML + LLMs

Trend-Eval is a hybrid machine learning pipeline that blends **traditional regression modeling** with **LLM-based sentiment and reasoning evaluation** to predict YouTube content performance (e.g., views per day).

## 🧠 Project Overview

This repository is designed to evaluate the impact of audio-visual and textual features—extracted from trailers, metadata, and user comments—on content success. It also benchmarks LLM reasoning capabilities using evaluation suites like MMLU and HellaSwag.

## 🔍 Key Features

- 📈 Predictive regression pipeline using `PyCaret` + `LightGBM`
- 🧮 Auto feature selection, normalization, and model tuning
- 🧠 LLM evaluation via Hugging Face on:
  - `hellaswag`, `mmlu`, `lambada`, `arc_challenge`, `winogrande`, `piqa`
- 📊 Visualization of model diagnostics (residuals, error, feature importance)
- 🔍 Advanced charting (Seaborn, Matplotlib)
- 📂 CSV export of predictions and metrics
- 🔎 Integrated with **Zeno** for advanced LLM visualization and comparison

---


## 🧪 Training Pipeline


## ⚙️ Automation Architecture

```mermaid
flowchart TD
  A[YouTube Video ID List] --> B[Fetch Metadata & Comments (YouTube API)]
  B --> C[Textual Analysis]
  C --> C1[Sentiment Analysis (Hugging Face)]
  C --> C2[Toxicity Detection]
  C --> C3[Zero-shot Classification (Genre, Budget, Clickbait)]
  
  B --> D[Download Video (pytube/yt-dlp)]
  D --> E[Audio Extraction]
  E --> E1[Speech-to-Text (wav2vec2)]
  E --> E2[Emotion, Speed, Speaker Count (speechbrain)]
  E --> E3[Audio Stats (librosa)]

  D --> F[Visual Processing (Thumbnails/Frames)]
  F --> F1[Captioning (BLIP-2)]
  F --> F2[Visual Classification (CLIP/OFA)]

  C1 --> G[Aggregate Features]
  C2 --> G
  C3 --> G
  E1 --> G
  E2 --> G
  E3 --> G
  F1 --> G
  F2 --> G

  G --> H[Final Feature Matrix (CSV/Parquet/DataFrame)]


The core `train.py` file runs a full PyCaret-based regression training pipeline:

- Loads and validates data from `data/raw/*.json`
- Cleans and filters data for modeling
- Applies multicollinearity reduction, variance filtering
- Trains a `LightGBM` model with `tune_model()` using Bayesian optimization
- Saves predictions, residuals, charts, and model artifacts in `charts/`

### 🗂 Output Files

- `charts/`: model diagnostic plots
- `charts/predictions.csv`: actual vs predicted
- `charts/metrics.csv`: performance metrics (R², RMSE, MAE)
- `model.pkl`: final model

---

## 🤖 LLM Benchmarking

The project also fine-tunes and evaluates lightweight LLMs on benchmark NLP tasks.

### 🔧 Models Used

```yaml
- nlptown/bert-base-multilingual-uncased-sentiment (max_length: 512)
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (max_length: 2048)
- meta-llama/Llama-3.2-1B (max_length: 2048)
```
