import pdfplumber
from langchain.schema import Document

def load_pdf(file_path):
    """Load a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return [Document(page_content=page) for page in text.strip().split("\n")]
