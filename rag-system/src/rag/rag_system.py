from typing import Dict

from src.generation.generator import Generator
from src.retrieval.retriever import Retriever


class RAGSystem:
    def __init__(self, model_name: str, api_key: str = None):
        self.retriever = Retriever()
        self.generator = Generator(model_name, api_key)

    def load_documents(self, file_path: str):
        self.retriever.load_documents(file_path)

    def add_document(self, content: str, metadata: Dict = None):
        self.retriever.add_document(content, metadata)

    def process_query(self, query: str, max_tokens: int = 150) -> str:
        retrieved_docs = self.retriever.retrieve(query)
        context = " ".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        return self.generator.generate(prompt, max_tokens)
