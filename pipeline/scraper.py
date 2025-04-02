import os
import json
from pytube import YouTube
from moviepy import VideoFileClip
from PIL import Image
import requests

# Scraping YouTube metadata

def scrape_youtube_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    return {
        "video_id": video_id,
        "title": yt.title,
        "description": yt.description,
        "views": yt.views,
        "publish_date": yt.publish_date.isoformat(),
        "length": yt.length,
        "author": yt.author,
        "keywords": yt.keywords,
        "thumbnail_url": yt.thumbnail_url
    }

# Scraping YouTube comments

def scrape_youtube_comments(video_id: str, api_key: str, max_comments: int = 100) -> list:
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": api_key,
        "maxResults": 100,
        "textFormat": "plainText"
    }
    while len(comments) < max_comments:
        response = requests.get(url, params=params).json()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        if "nextPageToken" in response:
            params["pageToken"] = response["nextPageToken"]
        else:
            break
    return comments[:max_comments]

# Downloading YouTube trailer

def download_youtube_trailer(video_id: str, output_dir: str = "downloads") -> str:
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension="mp4", progressive=True).order_by('resolution').desc().first()
    output_path = stream.download(output_path=output_dir, filename=f"{video_id}.mp4")
    return output_path

# Extracting audio from video

def extract_audio_from_video(video_path: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = video_path.replace(".mp4", ".wav")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path, codec='pcm_s16le')
    return output_path

# Extracting thumbnail

def extract_thumbnail(video_path: str, timestamp: float = 1.0) -> str:
    thumbnail_path = video_path.replace(".mp4", "_thumbnail.jpg")
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(timestamp)
    Image.fromarray(frame).save(thumbnail_path)
    return thumbnail_path

# Load video ids from external file

def load_video_ids_from_file(filepath: str) -> list:
    with open(filepath, "r") as f:
        return json.load(f)

# Process videos

def process_video_list(video_ids: list, api_key: str, output_dir: str = "downloads"):
    os.makedirs(output_dir, exist_ok=True)
    for video_id in video_ids:
        print(f"Processing video ID: {video_id}")
        # Scraping metadata and comments
        metadata = scrape_youtube_metadata(video_id)
        comments = scrape_youtube_comments(video_id, api_key)
        # Saving metadata and comments
        with open(f"{output_dir}/{video_id}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        with open(f"{output_dir}/{video_id}_comments.json", "w") as f:
            json.dump(comments, f, indent=2)
        # Downloading video, audio, and thumbnail
        video_path = download_youtube_trailer(video_id, output_dir)
        audio_path = extract_audio_from_video(video_path)
        thumbnail_path = extract_thumbnail(video_path)
        print(f"Saved: {video_path}, {audio_path}, {thumbnail_path}")

if __name__ == "__main__":
    video_ids = load_video_ids_from_file("video_ids.json")
    api_key = "YOUR_YOUTUBE_API_KEY"
    process_video_list(video_ids, api_key)