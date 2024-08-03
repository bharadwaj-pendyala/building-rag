"""
Text Chunking Techniques and Similarity Search for LLM Applications

This module provides various text chunking methods commonly used in Language Model (LLM) applications,
as well as information on similarity search techniques.

Similarity Search Techniques:
- Euclidean distance
- Cosine similarity
- Dot product similarity
Note: Best Similarity Metric: Match it to the one used to train your embedding model

Chunking Strategies for LLM Applications:
1. Embedding short (sentence) vs long content (document or paragraph)
   - The index may also be non-homogeneous and contain embeddings for chunks of varying sizes.

2. Chunking Considerations:
   - Nature of content being indexed
   - Embedding model being used and what chunk sizes it performs optimally
   - Expectations of length and complexity of user queries
   - Retrieved results usage - will they be used for semantic search, summarization etc.

3. Chunking Methods:
   a. Fixed Size Chunking: Choose token size and overlap between them
   b. "Content-aware" Chunking:
      - Sentence Splitting
        * Naive Splitting: Splitting sentences based on .
        * NLTK: Splits the text into sentences, creating meaningful chunks
        * spaCy: Sophisticated sentence segmentation (better context preservation)
   c. Recursive Chunking
      - Desired size/structure achieved through recursively calling itself
   d. Specialized Chunking
      - Markdown & LaTeX
   e. Semantic Chunking
      - Using embedding to extract semantic meaning present in data
        * Break up doc into sentences
        * Create sentence groups (Create groups using before and after for each)
          - Essentially all groups will be associated with an anchor sentence
        * Generate embeddings for each sentence and associate them to their anchor sentence
        * Compare distances between each group sequentially (lesser the better)

4. Figuring out best chunk size for your app:
   a. Preprocessing Data
   b. Selecting range of chunk sizes
      - Objective is to find a balance between preserving context and maintaining accuracy
   c. Evaluating the Performance of Each Chunk Size
      - Creating a single index or multiple indices

This module implements several of these chunking methods and provides utility functions for text processing.
"""

from typing import List, Dict, Optional

from langchain_text_splitters import (
    NLTKTextSplitter,
    SpacyTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    LatexTextSplitter,
    MarkdownHeaderTextSplitter,
)

def fixed_size_chunking(
    text: str, chunk_size: int = 256, chunk_overlap: int = 20
) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.

    This method implements the Fixed Size Chunking strategy mentioned in the module docstring.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def naive_split(text: str, delimiter: str = ".") -> List[str]:
    """
    Split text using a simple delimiter.

    This method implements the Naive Splitting strategy mentioned in the module docstring.

    Args:
        text (str): The input text to be split.
        delimiter (str): The character to use as a splitting point.

    Returns:
        List[str]: A list of text chunks.
    """
    return [chunk.strip() for chunk in text.split(delimiter) if chunk.strip()]


def split_text_with_nltk(text: str) -> List[str]:
    """
    Split text using NLTK (Natural Language Toolkit).

    This method implements the NLTK splitting strategy mentioned in the module docstring.

    Args:
        text (str): The input text to be split.

    Returns:
        List[str]: A list of sentences.
    """
    return NLTKTextSplitter().split_text(text)


def split_text_with_spacy(text: str) -> List[str]:
    """
    Split text using spaCy.

    This method implements the spaCy splitting strategy mentioned in the module docstring.

    Args:
        text (str): The input text to be split.

    Returns:
        List[str]: A list of sentences.
    """
    return SpacyTextSplitter().split_text(text)


def split_text_with_recursive(
    text: str, chunk_size: int = 256, chunk_overlap: int = 20
) -> List[Dict[str, str]]:
    """
    Split text using recursive character text splitter.

    This method implements the Recursive Chunking strategy mentioned in the module docstring.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The target size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the split text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.create_documents([text])


def split_text_with_markdown(
    text: str, chunk_size: int = 100, chunk_overlap: int = 0
) -> List[Dict[str, str]]:
    """
    Split markdown text.

    This method implements the Specialized Chunking strategy for Markdown mentioned in the module docstring.

    Args:
        text (str): The markdown text to be split.
        chunk_size (int): The target size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the split markdown text.
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


def split_text_with_markdown_header(
    text: str,
    chunk_size: int = 100,
    chunk_overlap: int = 0,
    headers_to_split_on: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Split markdown text by headers.

    This method implements the Header-Based Chunking strategy for Markdown mentioned in the module docstring.

    Args:
        text (str): The markdown text to be split.
        chunk_size (int): The target size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the split markdown text.
    """
    if headers_to_split_on is None:
        headers_to_split_on = ["#", "##", "###"]  # Default headers to split on

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = splitter.split_text(text)
    return md_header_splits


def split_text_with_latex(
    text: str, chunk_size: int = 100, chunk_overlap: int = 0
) -> List[Dict[str, str]]:
    """
    Split LaTeX text.

    This method implements the Specialized Chunking strategy for LaTeX mentioned in the module docstring.

    Args:
        text (str): The LaTeX text to be split.
        chunk_size (int): The target size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the split LaTeX text.
    """
    splitter = LatexTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


def print_chunks(chunks: List[str | Dict[str, str]], method_name: str):
    """
    Print chunks with method name.

    This utility function is useful for displaying the results of different chunking methods.

    Args:
        chunks (List[str | Dict[str, str]]): The list of chunks to print.
        method_name (str): The name of the chunking method used.
    """
    print(f"\n{method_name} Results:")
    for i, chunk in enumerate(chunks, 1):
        if isinstance(chunk, dict):
            print(f"Chunk {i}:\n{chunk['text']}\n")
        else:
            print(f"Chunk {i}:\n{chunk}\n")


# Example usage
if __name__ == "__main__":
    api_key = authenticate_pinecone()
    set_openai_api_key()

    sample_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."""

    sample_mrkdwn = """# Heading 1
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

    ## Heading 2
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
    nisi ut aliquip ex ea commodo consequat.

    ### Heading 3
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

    ## Another Heading 2
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
    deserunt mollit anim id est laborum."""

    sample_latex = r"""
    \documentclass{article}
    \begin{document}

    \title{Sample LaTeX Document}
    \author{Author Name}
    \date{\today}
    \maketitle

    \section{Introduction}
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

    \section{Methods}
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
    nisi ut aliquip ex ea commodo consequat.

    \section{Results}
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

    \section{Conclusion}
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
    deserunt mollit anim id est laborum.

    \end{document}
    """

    print_chunks(
        fixed_size_chunking(sample_text, chunk_size=50, chunk_overlap=10),
        "Fixed Size Chunking",
    )
    print_chunks(naive_split(sample_text), "Naive Splitting")
    print_chunks(split_text_with_nltk(sample_text), "NLTK Splitter")
    print_chunks(split_text_with_spacy(sample_text), "spaCy Splitter")
    print_chunks(split_text_with_recursive(sample_text), "Recursive Splitter")
    print_chunks(split_text_with_markdown(sample_mrkdwn), "Markdown Splitter")
    print_chunks(split_text_with_latex(sample_latex), "LaTeX Splitter")
    print_chunks(
        split_text_with_markdown_header(sample_mrkdwn), "Markdown Header Splitter"
    )
