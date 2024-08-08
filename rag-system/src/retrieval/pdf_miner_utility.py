from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import os

# Load the model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)  # You can choose other models from sentence-transformers


def get_embeddings(text):
    # Generate embeddings using the sentence-transformers model
    return model.encode(text)


current_directory = os.getcwd()
doc_location = os.path.join(
    current_directory, "rag-system", "src", "retrieval", "insurance_checklist.pdf"
)
text = extract_text(doc_location)
chunks = [text[i : i + 500] for i in range(0, len(text), 500)]  # split text into chunks

for i, chunk in enumerate(chunks):
    embedding = get_embeddings(chunk)
    print(f"embeddings for reference: {embedding}")
