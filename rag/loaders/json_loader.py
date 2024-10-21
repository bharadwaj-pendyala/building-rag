import json
from langchain.schema import Document

def flatten_json(data, prefix=''):
    """Recursively flatten a JSON object."""
    flattened = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            flattened.extend(flatten_json(value, new_prefix))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_prefix = f"{prefix}[{i}]"
            flattened.extend(flatten_json(item, new_prefix))
    else:
        flattened.append((prefix, str(data)))
    return flattened

def load_json(file_path, chunk_size=500):
    """Load a JSON file and return a list of Document objects."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Flatten the JSON structure
    flattened_data = flatten_json(data)
    
    # Convert flattened data to string format
    content = "\n".join([f"{key}: {value}" for key, value in flattened_data])
    
    # Split content into chunks
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    # Create Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents