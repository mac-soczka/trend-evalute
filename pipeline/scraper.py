import os
import json
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
OUTPUT_FILE = "video_ids.json"

def fetch_video_ids(query="official movie trailer", max_results=1000):
    """
    Fetches a list of YouTube video IDs based on a search query.
    """
    if not API_KEY:
        raise EnvironmentError("Missing YOUTUBE_API_KEY in .env")

    url = "https://www.googleapis.com/youtube/v3/search"
    video_ids = []
    page_token = None

    print(f"Searching YouTube for: '{query}'")

    while len(video_ids) < max_results:
        params = {
            "part": "id",
            "q": query,
            "type": "video",
            "maxResults": 50,
            "key": API_KEY,
        }

        if page_token:
            params["pageToken"] = page_token

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            if item["id"]["kind"] == "youtube#video":
                video_ids.append(item["id"]["videoId"])

        page_token = data.get("nextPageToken")
        if not page_token:
            print("No more pages available.")
            break

        print(f"Collected: {len(video_ids)} video IDs so far...")

    return video_ids[:max_results]

if __name__ == "__main__":
    print("Fetching YouTube video IDs...")
    video_ids = fetch_video_ids(query="official movie trailer", max_results=1000)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(video_ids, f, indent=2)

    print(f"✅ Saved {len(video_ids)} video IDs to '{OUTPUT_FILE}'")
