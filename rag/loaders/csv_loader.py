import pandas as pd
from langchain.schema import Document

def load_csv(file_path, chunk_size=500):
    """Load a CSV file and return a list of Document objects."""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert DataFrame to list of dictionaries
    records = df.to_dict(orient='records')
    
    # Convert records to string format
    content = ""
    for record in records:
        content += "Row:\n"
        for key, value in record.items():
            content += f"{key}: {value}\n"
        content += "\n"
    
    # Split content into chunks
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    # Create Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents