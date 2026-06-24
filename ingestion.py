from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langsmith import traceable
from config import embeddings


@traceable(name="load_pdf")
def load_pdf(pdf_path: str,):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {pdf_path}")
    return documents 


@traceable(name="split_documents")
def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    return chunks

@traceable(name="create_vectorstore")
def create_vectorstore(chunks):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("FAISS Created")

    return vectorstore

@traceable(name="save_vectorstore")
def save_vectorstore(vectorstore):

    vectorstore.save_local(
        "vectorstore/faiss_index"
    )

    print("Vector Store Saved")


@traceable(name="Load_from_vectorstore")
def load_vectorstore():
    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
     )
    print("Vector Store Loaded")
    return vectorstore

