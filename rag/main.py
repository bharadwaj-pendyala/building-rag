import os

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config.settings import OPENAI_API_KEY
from rag.loaders import load_csv, load_json, load_pdf, load_txt, load_youtube_transcript
from rag.utils import split_text


def build_qa_chain() -> RetrievalQA:
    """Build and return the RetrievalQA chain used by the CLI."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    text_file = os.path.join(data_dir, "zephyria_ecosystems.txt")
    json_file = os.path.join(data_dir, "galactic_civilization.json")
    pdf_file = os.path.join(data_dir, "lumina_language.pdf")
    csv_file = os.path.join(data_dir, "exoplanet_atmospheres.csv")
    youtube_video_url = "https://www.youtube.com/watch?v=7ATtD8x7vV0"  # crash course on astronomy

    txt_data = load_txt(text_file)
    json_data = load_json(json_file)
    pdf_data = load_pdf(pdf_file)
    csv_data = load_csv(csv_file)
    youtube_data = load_youtube_transcript(youtube_video_url)

    all_texts = txt_data + json_data + pdf_data + csv_data + youtube_data
    texts = split_text(all_texts)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    docsearch = Chroma.from_documents(texts, embeddings)

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def run_cli() -> None:
    """Run the interactive Q&A loop."""
    qa = build_qa_chain()

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        response = qa({"query": query})
        print("Response:", response["result"])


if __name__ == "__main__":
    run_cli()
