from langchain.text_splitter import CharacterTextSplitter

def split_text(texts, chunk_size=1000, chunk_overlap=0):
    """Splits text into manageable chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(texts)
