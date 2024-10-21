from loaders import load_txt, load_pdf, load_json, load_csv, load_youtube_transcript
from utils import split_text
from config.settings import OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# File paths for different file types
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

text_file = os.path.join(data_dir, "star_wars_excerpt.txt")
json_file = os.path.join(data_dir, "star_wars_excerpt.json")
pdf_file = os.path.join(data_dir, "star_wars_excerpt.pdf")
csv_file = os.path.join(data_dir, "star_wars_excerpt.csv")
youtube_video_url = "https://www.youtube.com/watch?v=_lOT2p_FCvA"

# Load all data
txt_data = load_txt(text_file)
json_data = load_json(json_file)
pdf_data = load_pdf(pdf_file)
csv_data = load_csv(csv_file)
youtube_data = load_youtube_transcript(youtube_video_url)

# Combine all text data into one list
all_texts = txt_data + json_data + pdf_data + csv_data + youtube_data

# Split the combined text into manageable chunks
texts = split_text(all_texts)

# Generate text embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Specify the OpenAI model to use
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create a VectorStore (Chroma) to store and retrieve embeddings
docsearch = Chroma.from_documents(texts, embeddings)

# Create a custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize the Retriever using the VectorStore
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# Loop for user queries
while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # Query the model with the user input
    response = qa({"query": query})

    print("Response:", response["result"])
