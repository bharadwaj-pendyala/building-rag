s# RAG System

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and OpenAI's models. The system loads various document types (text, PDF, JSON, CSV, and YouTube transcripts), splits them into chunks, generates embeddings, and allows users to ask questions about the content.

## Project Architecture

![RAG Arch Overview.](docs/diagrams/RAG_Arch.png)

While the current implementation may not exactly match this architecture, this diagram provides a general overview of a RAG system's components and flow.

## Features

- **Document Loading**: Load text, PDF, JSON, CSV, and YouTube transcripts.
- **Text Splitting**: Manageable chunks for efficient processing.
- **Text Embeddings**: Generated using OpenAI’s models.
- **Vector Storage**: Using Chroma for fast retrieval.
- **Question Answering**: Powered by OpenAI’s GPT-3.5-turbo model.
- **Interactive CLI**: Command-line interface for querying the system.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

## Project Structure

```
rag_project/
│
├── loaders/
│   ├── text_loader.py     # For loading text files
│   ├── pdf_loader.py      # For loading PDFs
│   ├── json_loader.py     # For loading JSON files
│   ├── csv_loader.py      # For loading CSV files
│   └── youtube_loader.py  # For loading YouTube transcripts
│
├── utils/
│   └── splitter.py        # Text splitting logic
│
├── config/
│   └── settings.py        # Configuration and environment variables
│
├── main.py                # Main entry point for running the system
├── requirements.txt       # Required Python libraries
└── .env                   # Environment variables (API keys, etc.)
```

## Usage

1. Place your files:
   - Add your documents (e.g., text, PDFs, JSON, CSV, or YouTube video links) in a designated folder, or specify the paths in `main.py`.

2. Run the script:
   ```bash
   python main.py
   ```

3. Enter your questions when prompted. Type `'exit'` to quit the program.

## How it Works

1. **Document Loading**: The script loads documents using loaders from the `loaders/` package, which supports text files, PDFs, JSON, CSV, and YouTube transcripts.
2. **Text Splitting**: Text is split into manageable chunks using the `split_text` function from the `utils/splitter.py`.
3. **Embeddings Generation**: OpenAI's embedding model is used to generate text embeddings.
4. **Vector Storage**: A Chroma vector store is created to store and retrieve these embeddings.
5. **Question-Answering**: A custom prompt template is used to structure how GPT-3.5-turbo generates answers.
6. **Interactive Querying**: Users can interactively ask questions about the content of the loaded documents.

## Example Loaders

- **Text File**: `load_txt(file_path)`
- **PDF File**: `load_pdf(file_path)`
- **JSON File**: `load_json(file_path)`
- **CSV File**: `load_csv(file_path)`
- **YouTube Transcript**: `load_youtube_transcript(video_url)`

## Customization

- Modify `chunk_size` and `chunk_overlap` in `split_text()` to adjust how documents are split.
- Change the `model_name` in `ChatOpenAI` initialization to use a different OpenAI model.
- Adjust the `prompt_template` in `main.py` to customize the system's responses.
- Add more loaders for other file types as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.