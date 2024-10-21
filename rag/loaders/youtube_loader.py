from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document

def load_youtube_transcript(video_url):
    """Load the transcript from a YouTube video using the video ID."""
    try:
        video_id = video_url.split("v=")[1]  # Extract the video ID from the URL
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript])
        return [Document(page_content=transcript_text)]
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return []
