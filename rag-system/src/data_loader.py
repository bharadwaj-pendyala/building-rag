import json


def load_documents(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Usage:
# documents = load_documents('data/documents.json')
