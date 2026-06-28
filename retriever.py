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

    docs = load_pdf("data/lora.pdf")

    chunks = split_documents(docs)

    return chunks

@traceable(name="faiss_retriever")
def get_faiss_retriever():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":15}
    )

    return retriever


@traceable(name="bm25_retriever")
def get_bm25_retriever():

    chunks = load_chunks()

    bm25 = BM25Retriever.from_documents(chunks)

    bm25.k = 15

    return bm25

@traceable(name="hybrid_retrieve")
def hybrid_retrieve(query: str):

    vectorstore = load_vectorstore()

    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 7}
    )

    bm25_retriever = get_bm25_retriever()
    bm25_retriever.k = 7

    faiss_docs = faiss_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    # --------------------------
    # Round-Robin Merge
    # --------------------------

    merged_docs = []
    seen = set()

    max_len = max(len(faiss_docs), len(bm25_docs))

    for i in range(max_len):

        if i < len(faiss_docs):
            doc = faiss_docs[i]
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged_docs.append(doc)

        if i < len(bm25_docs):
            doc = bm25_docs[i]
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged_docs.append(doc)

    
    # Return top candidates
    return merged_docs[:10]