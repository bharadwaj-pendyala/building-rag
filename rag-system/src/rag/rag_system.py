from typing import List, Dict
from .retriever import EnhancedRetriever
from .generator import EnhancedGenerator


class EnhancedRAGSystem:
    def __init__(self, model_name: str, api_key: str = None):
        self.retriever = EnhancedRetriever()
        self.generator = EnhancedGenerator(model_name, api_key)

    def load_documents(self, file_path: str):
        self.retriever.load_documents(file_path)

    def add_document(self, content: str, metadata: Dict = None):
        self.retriever.add_document(content, metadata)

    def process_query(self, query: str, max_tokens: int = 150) -> str:
        retrieved_docs = self.retriever.retrieve(query)
        context = " ".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        return self.generator.generate(prompt, max_tokens)
