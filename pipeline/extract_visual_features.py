import os
import json
import torch
import requests
import pandas as pd
from PIL import Image
from typing import List, Dict
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ================================
# 1️⃣ Load BLIP-2 Model
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Salesforce/blip2-opt-2.7b"

processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# ================================
# 2️⃣ Utility Functions
# ================================
def fetch_thumbnail(video_id: str) -> Image.Image:
    url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    try:
        return Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except Exception:
        raise ValueError(f"❌ Could not fetch thumbnail for {video_id}")

def ask_blip2(image: Image.Image, prompt: str) -> str:
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()

def extract_features(video_id: str) -> Dict:
    try:
        image = fetch_thumbnail(video_id)
    except Exception as e:
        print(str(e))
        return {"video_id": video_id, "error": "no_thumbnail"}

    features = {"video_id": video_id}

    try:
        features["thumbnail_caption"] = ask_blip2(image, "Describe the scene in detail.")

        actor_answer = ask_blip2(image, "How many recognizable actors are visible?")
        features["num_known_actors"] = int(next((w for w in actor_answer.split() if w.isdigit()), 0))
        features["has_celebrity_actor"] = features["num_known_actors"] > 0

        genre_answer = ask_blip2(image, "What genre does this image most likely represent?")
        features["genre"] = genre_answer

        budget_answer = ask_blip2(image, "Does this look like a low, medium, or high budget production?")
        features["budget_estimate"] = next((x for x in ["low", "medium", "high"] if x in budget_answer.lower()), "unknown")

        clickbait_answer = ask_blip2(image, "Does this thumbnail look like clickbait?")
        features["clickbait_thumbnail"] = "yes" in clickbait_answer.lower()

        mood_answer = ask_blip2(image, "What is the overall mood or tone of this image?")
        features["visual_mood"] = mood_answer

    except Exception as e:
        features["error"] = str(e)

    return features

# ================================
# 3️⃣ Load JSON Video IDs
# ================================
def load_video_ids_from_json(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON should be a list of video IDs.")
    return [vid.strip() for vid in data if isinstance(vid, str) and vid.strip()]

# ================================
# 4️⃣ Main Runner
# ================================
def run_batch(json_path: str, output_csv="blip2_features.csv"):
    video_ids = load_video_ids_from_json(json_path)
    results = []
    for vid in video_ids:
        print(f"🔍 Processing {vid}...")
        result = extract_features(vid)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ All done. Saved to {output_csv}")

# ================================
# 5️⃣ Entry Point
# ================================
if __name__ == "__main__":
    input_json = "video_ids.json"  # Update this if needed
    run_batch(input_json)
