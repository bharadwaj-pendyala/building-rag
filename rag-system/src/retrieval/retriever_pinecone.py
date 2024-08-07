from .storage_utility import docsearch, pc, index_name
from langchain_openai import ChatOpenAI
import langchain.chains as lc_chains
import os

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.0,
)

qa = lc_chains.RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

query1 = "What are the first 3 steps for getting started with the WonderVector5000?"

query2 = "The Neural Fandango Synchronizer is giving me a headache. What do I do?"


query1_with_knowledge = qa.invoke(query1)
query1_without_knowledge = llm.invoke(query1)

print(query1_with_knowledge)
print()
print(query1_without_knowledge)


query2_with_knowledge = qa.invoke(query2)
query2_without_knowledge = llm.invoke(query2)

print(query2_with_knowledge)
print()
print(query2_without_knowledge)

pc.delete_index(index_name)
