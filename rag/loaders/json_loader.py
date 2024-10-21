import json
from langchain.schema import Document

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return [Document(page_content=data["scene"])] + [Document(page_content=dialogue) for char in data["characters"] for dialogue in char["dialogue"]]
