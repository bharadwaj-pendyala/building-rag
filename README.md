# Enhanced RAG System

This project implements an enhanced Retrieval-Augmented Generation (RAG) system from scratch. It combines the power of retrieval-based and generation-based approaches to create more accurate and contextually relevant responses, with support for various document types and language models.

## Project Structure

```
rag-system/
├── src/
│   ├── retrieval/
│   │   └── enhanced_retriever.py
│   ├── generation/
│   │   └── enhanced_generator.py
│   ├── rag/
│   │   └── enhanced_rag_system.py
│   └── main.py
├── tests/
│   └── test_rag_system.py
├── data/
├── requirements.txt
├── README.md
└── .gitignore
```

## Features

- Enhanced retrieval component supporting JSON, TXT, and PDF inputs
- Flexible generation component supporting multiple language models (GPT-3.5, GPT-4, Gemini, Hugging Face models)
- Integration of retrieval and generation for enhanced output
- Simple API for interacting with the RAG system, including file upload and document addition

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/bharadwaj-pendyala/building-rag.git
   cd rag-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up API keys:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

1. Start the API:
   ```
   uvicorn src.main:app --reload
   ```

2. Use the API endpoints:

   a. Upload a document:
   ```
   curl -X POST -F "file=@path/to/your/file.pdf" http://localhost:8000/upload
   ```

   b. Add a document directly:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"content":"Your document content here"}' http://localhost:8000/add_document
   ```

   c. Query the RAG system:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"text":"Your query here"}' http://localhost:8000/query
   ```

## Testing Locally

1. Prepare test documents:
   Place some test documents (JSON, TXT, PDF) in the `data/` directory.

2. Update the `src/main.py` file:
   - Set the desired language model and API key:
     ```python
     rag_system = EnhancedRAGSystem(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
     ```
   - Load some initial documents:
     ```python
     rag_system.load_documents("data/your_test_document.pdf")
     ```

3. Start the API server:
   ```
   uvicorn src.main:app --reload
   ```

4. Use a tool like curl or Postman to test the endpoints:

   a. Upload a new document:
   ```
   curl -X POST -F "file=@data/new_document.pdf" http://localhost:8000/upload
   ```

   b. Add a document directly:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"content":"This is a test document content."}' http://localhost:8000/add_document
   ```

   c. Query the system:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"text":"What information can you provide about the uploaded documents?"}' http://localhost:8000/query
   ```

5. Check the responses to ensure the system is working as expected.

## Running Tests

Run the tests using pytest:
```
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.