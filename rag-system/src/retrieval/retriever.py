import json
import os
from typing import List, Dict

import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self):
        self.documents = []
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.document_embeddings = None

    def add_document(self, content: str, metadata: Dict = None):
        doc = {"content": content, "metadata": metadata or {}}
        self.documents.append(doc)
        self._update_embeddings()

    def load_documents(self, file_path: str):
        _, ext = os.path.splitext(file_path)
        if ext == '.json':
            self._load_json(file_path)
        elif ext == '.txt':
            self._load_text(file_path)
        elif ext == '.pdf':
            self._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        self._update_embeddings()

    def _load_json(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        for item in data:
            self.add_document(item['content'], item.get('metadata'))

    def _load_text(self, file_path: str):
        with open(file_path, 'r') as f:
            content = f.read()
        self.add_document(content, {"source": file_path})

    def _load_pdf(self, file_path: str):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
        self.add_document(content, {"source": file_path})

    def _update_embeddings(self):
        self.document_embeddings = self.model.encode([doc['content'] for doc in self.documents])

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
