import os
import googleapiclient.discovery

def get_video_ids_by_topic(api_key: str, topic: str, max_results: int = 50):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    request = youtube.search().list(
        part="id",
        q=topic,
        type="video",
        maxResults=min(max_results, 50),
        relevanceLanguage="en",
        safeSearch="moderate"
    )

    video_ids = []
    response = request.execute()

    for item in response["items"]:
        if item["id"]["kind"] == "youtube#video":
            video_ids.append(item["id"]["videoId"])

    return video_ids

if __name__ == "__main__":
    API_KEY = os.environ.get("YOUTUBE_API_KEY")  # Set this in your environment
    topic = "movie trailers 2024"
    max_results = 30

    if not API_KEY:
        raise EnvironmentError("Set the YOUTUBE_API_KEY environment variable.")

    video_ids = get_video_ids_by_topic(API_KEY, topic, max_results)
    print(f"Found {len(video_ids)} video IDs:")
    print(video_ids)
