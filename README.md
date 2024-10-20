# RAG System

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and OpenAI's models. The system loads a text document, splits it into chunks, generates embeddings, and allows users to ask questions about the content.

## Project Architecture

![RAG Arch Overview.](docs/diagrams/RAG_Arch.png)

While the current implementation may not exactly match this architecture, this diagram provides a general overview of a RAG system's components and flow.

## Features

- Document loading and text splitting
- Text embedding generation using OpenAI
- Vector storage using Chroma
- Question answering using OpenAI's GPT-3.5-turbo model
- Interactive command-line interface for querying the system

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install langchain langchain_community langchain_openai chromadb python-dotenv
   ```

4. Set up your OpenAI API key:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Prepare your document:
   Place your text document (e.g., `dummy.txt`) in the same directory as the script.

2. Run the script:
   ```
   python rag_system.py
   ```

3. Enter your questions when prompted. Type 'exit' to quit the program.

## How it Works

1. The script loads the document using `TextLoader`.
2. The text is split into manageable chunks using `CharacterTextSplitter`.
3. Text embeddings are generated using OpenAI's embedding model.
4. A Chroma vector store is created to store and retrieve the embeddings.
5. A custom prompt template is defined for the question-answering process.
6. The system uses OpenAI's GPT-3.5-turbo model to generate answers based on the retrieved context.
7. Users can interactively ask questions about the document's content.

## Customization

- You can modify the `chunk_size` and `chunk_overlap` parameters in the `CharacterTextSplitter` to adjust how the document is split.
- Change the `model_name` in the `ChatOpenAI` initialization to use a different OpenAI model.
- Adjust the prompt template to customize how the system generates answers.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.