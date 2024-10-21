import os
from dotenv import load_dotenv

load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("API key not found. Please set 'OPENAI_API_KEY' in the environment variables.")
