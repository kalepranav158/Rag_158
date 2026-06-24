from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings



llm = ChatOllama(model="qwen2.5:3b",temperature=0.7, max_tokens=512, top_k=10, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
