import pytest
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.rag.rag_system import RAGSystem


@pytest.fixture
def sample_documents():
    return [
        {"id": "1", "content": "The quick brown fox jumps over the lazy dog."},
        {"id": "2", "content": "Python is a versatile programming language."},
        {
            "id": "3",
            "content": "Machine learning is a subset of artificial intelligence.",
        },
    ]


def test_retriever(sample_documents):
    retriever = Retriever(sample_documents)
    results = retriever.retrieve("Python programming")
    assert len(results) == 3
    assert results[0]["id"] == "2"


def test_generator():
    generator = Generator()
    result = generator.generate("Hello, world!")
    assert len(result) > 0


def test_rag_system(sample_documents):
    rag = RAGSystem(sample_documents)
    result = rag.process_query("Tell me about Python")
    assert len(result) > 0
    assert "Python" in result


# Run with: pytest tests/test_rag_system.py
