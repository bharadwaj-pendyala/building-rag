from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Error handling in case API key is not available
if not api_key:
    raise ValueError(
        "API key not found. Please set 'OPENAI_API_KEY' in the environment variables."
    )

# Load the document
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "dummy.txt")
loader = TextLoader(file_path)
data = loader.load()

# Split the document into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print("Split texts:", texts)

# Generate text embeddings
embeddings = OpenAIEmbeddings()

print("Embeddings object:", embeddings)

# Specify the OpenAI model to use
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create a VectorStore (Chroma) to store and retrieve embeddings
docsearch = Chroma.from_documents(texts, embeddings)

# Create a custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

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
    # Ask the user for a question
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # Query the model with the user input
    response = qa({"query": query})

    print("Response:", response["result"])
