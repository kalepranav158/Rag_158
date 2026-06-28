from ingestion import (
    load_pdf,
    split_documents,
    create_vectorstore,
    save_vectorstore,
)

PDF_PATH = "data/lora.pdf"

docs = load_pdf(PDF_PATH)

chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

save_vectorstore(vectorstore)

print("\nVector Store Ready ✅")