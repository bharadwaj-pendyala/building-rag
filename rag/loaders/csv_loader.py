import pandas as pd
from langchain.schema import Document

def load_csv(file_path):
    """Load a CSV file."""
    df = pd.read_csv(file_path)
    return [Document(page_content=row) for row in df["Dialogue"].tolist()]
