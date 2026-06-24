# retriever.py

from langsmith import traceable

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from config import embeddings
from ingestion import load_pdf, split_documents


@traceable(name="load_vectorstore")
def load_vectorstore():

    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Vector Store Loaded")

    return vectorstore


@traceable(name="load_chunks")
def load_chunks():

    docs = load_pdf("data/one.pdf")

    chunks = split_documents(docs)

    return chunks


@traceable(name="faiss_retriever")
def get_faiss_retriever():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":5}
    )

    return retriever


@traceable(name="bm25_retriever")
def get_bm25_retriever():

    chunks = load_chunks()

    bm25 = BM25Retriever.from_documents(chunks)

    bm25.k = 5

    return bm25

@traceable(name="hybrid_retrieve")
def hybrid_retrieve(query: str):

    vectorstore = load_vectorstore()

    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k":5}
    )

    bm25_retriever = get_bm25_retriever()

    faiss_docs = faiss_retriever.invoke(query)

    bm25_docs = bm25_retriever.invoke(query)

    all_docs = faiss_docs + bm25_docs

    unique_docs = []

    seen = set()

    for doc in all_docs:

        if doc.page_content not in seen:

            seen.add(doc.page_content)

            unique_docs.append(doc)

    return unique_docs[:5]