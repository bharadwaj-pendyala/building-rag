from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.document_embeddings = self.model.encode(
            [doc["content"] for doc in documents]
        )

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
