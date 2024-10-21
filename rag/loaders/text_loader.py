from langchain_community.document_loaders import TextLoader

def load_txt(file_path):
    """Load a text file."""
    loader = TextLoader(file_path)
    return loader.load()
