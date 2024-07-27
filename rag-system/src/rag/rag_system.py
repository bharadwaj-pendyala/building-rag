from ..retrieval.retriever import Retriever
from ..generation.generator import Generator


class RAGSystem:
    def __init__(self, documents):
        self.retriever = Retriever(documents)
        self.generator = Generator()

    def process_query(self, query, max_length=150):
        retrieved_docs = self.retriever.retrieve(query)
        context = " ".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        return self.generator.generate(prompt, max_length)
