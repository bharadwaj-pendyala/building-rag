from fastapi import FastAPI
from pydantic import BaseModel
from src.rag.rag_system import RAGSystem

app = FastAPI()

# Initialize your documents and RAG system here
documents = [
    {
        "id": "1",
        "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.",
    },
    {
        "id": "2",
        "content": "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    },
    # Add more documents as needed
]

rag_system = RAGSystem(documents)


class Query(BaseModel):
    text: str


@app.post("/query")
async def process_query(query: Query):
    result = rag_system.process_query(query.text)
    return {"result": result}


# Run with: uvicorn src.main:app --reload
