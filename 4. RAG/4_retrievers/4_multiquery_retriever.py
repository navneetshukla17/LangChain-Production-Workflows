# necessary imports
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

# load huggingface api key
load_dotenv()

# sample documents
docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source":"H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source":"H3"}),
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H4"}),
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H5"}),
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H6"}),
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H7"}),
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H8"}),
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H9"}),
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source":"H10"})
]

# embedding model - for multi query retriever
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# LLM config
llm = ChatGroq(model="llama-3.1-8b-instant")

# pinecone vector store
vector_store = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embedding_model,
    index_name=os.getenv("PINECONE_INDEX_NAME")
)
# base retriever for multi query retriever
base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

# multi query retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=base_retriever
)

# similarity search retriever
similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2}
)

# user query & relevant context
query = "how to improve energy level and improve balance"


# final output
result1 = multi_query_retriever.invoke(input=query)
for i, res in enumerate(result1, start=1):
    print("Results based on Multi Query Retriever:")
    print(f"\nPage: {i}\n")
    print(f"{res.page_content}")

result2 = similarity_retriever.invoke(input=query)
for i, res in enumerate(result1, start=1):
    print("Results based on Similarity:")
    print(f"\nPage: {i}\n")
    print(f"{res.page_content}")






