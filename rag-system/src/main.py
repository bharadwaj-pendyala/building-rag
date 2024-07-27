from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from src.rag.enhanced_rag_system import EnhancedRAGSystem
from dotenv import load_dotenv
import os

app = FastAPI()

# Initialize your RAG system here
rag_system = EnhancedRAGSystem(
    model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY")
)

class Query(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: Query):
    result = rag_system.process_query(query.text)
    return {"result": result}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    rag_system.load_documents(file_path)
    return {"message": f"File {file.filename} uploaded and processed successfully"}


@app.post("/add_document")
async def add_document(content: str):
    rag_system.add_document(content)
    return {"message": "Document added successfully"}


# Run with: uvicorn src.main:app --reload
