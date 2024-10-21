# RAG System

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

## Sample Prompts for Included Test Data

### Text File

1. What is the diameter of Zephyria compared to Earth?
2. How long is a Zephyrian year in Earth days?
3. What are the four major ecosystems on Zephyria?
4. What is the dominant species in the Luminous Forests?
5. How tall can Quartz spires in the Crystal Deserts grow?
6. What causes the Floating Islands to levitate?
7. What is unique about the water in Zephyria's Geothermal Seas?
8. Which ecosystem has the highest Zephyrian Biodiversity Index (ZBI) score?
9. Describe the symbiotic relationship in the Luminous Forests.
10. What are two environmental challenges facing Zephyria?

### CSV

1. Which planet has the highest habitability score?
2. What is the atmospheric pressure on Glacius-9?
3. List all planets with a primary atmosphere of Nitrogen.
4. What are the secondary elements in Terraxa Prime's atmosphere?
5. Which planet is closest to Earth in terms of atmospheric pressure?

### JSON

1. What is the name of the Zephyrian Collective's home system?
2. How many species are part of the Zephyrian Collective?
3. What is the maximum speed of their FTL travel method?
4. Who are the Zephyrian Collective's allies?
5. What was their most recent notable achievement, and in what year did it occur?

### PDF

1. How do the Luminans communicate instead of using sound?
2. What are the five basic light frequencies used in the Lumina language?
3. How is verb tense indicated in Lumina?
4. What is the word for "Peace" in Lumina?
5. Describe the cultural significance of the Lumina language.

### YouTube Transcript

1. What are the main methods used to detect exoplanets mentioned in the video?
2. How does the transit method of detecting exoplanets work?
3. What is the radial velocity method, and how does it help in detecting exoplanets?
4. According to the video, what was special about the discovery of 51 Pegasi b?
5. What are some of the challenges in directly imaging exoplanets?
6. How do astronomers determine the composition of exoplanet atmospheres?
7. What is the habitable zone, and why is it important in the search for exoplanets?
8. What are some of the unexpected types of planets that have been discovered?
9. How has the discovery of exoplanets changed our understanding of planetary formation?
10. What future missions or technologies does the video mention for studying exoplanets?

### Integrative Prompts

1. Compare the exoplanet detection methods described in the Crash Course video with the data we have on fictional planets in our CSV file. How might these methods have been used to gather our fictional data?
2. Based on the information from the Crash Course video about exoplanet atmospheres, analyze the atmospheric compositions in our CSV file. Which of our fictional planets might be considered most Earth-like?
3. The video mentions hot Jupiters and super-Earths. How do these compare to the planets described in our fictional datasets?
4. Using the concepts from the Crash Course video, how might the Zephyrian Collective from our JSON data have developed their FTL travel technology to overcome the vast distances between stars?
5. The video discusses the potential for finding life on exoplanets. How might this relate to the development of the Lumina language described in our PDF data?

## Customization

- Modify `chunk_size` and `chunk_overlap` in `split_text()` to adjust how documents are split.
- Change the `model_name` in `ChatOpenAI` initialization to use a different OpenAI model.
- Adjust the `prompt_template` in `main.py` to customize the system's responses.
- Add more loaders for other file types as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.