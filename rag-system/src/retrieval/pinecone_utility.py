"""
# Similarity Search Techniques:
# Euclidean distance,
# cosine similarity and
# dot product similarity
# Note: Best Similarity Metric: To use match it to the one used to train your embedding model
"""

import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone_notebooks.colab import Authenticate

# initialize connection to pinecone (orget API key at app.pinecone.io)
if not os.environ.get("PINECONE_API_KEY"):

    Authenticate()


api_key = os.environ.get("PINECONE_API_KEY")

# available at platform.openai.com/api-keys
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")


"""
# Chunking Strategies for LLM Applications
# Embedding short(sentence) vs long content(document or paragraph)
"""

# Use a text splitter to split the pdf into chunks and store in pinecone DB
markdown_document = ""

headers_to_split_on = [("##", "Header 2")]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

print(md_header_splits)
