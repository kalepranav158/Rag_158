#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
#llm = ChatGoogleGenerativeAI(
#   model="gemini-2.5-flash",
#    temperature=0
#)

judge_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
    max_tokens=256,
)

generator_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2,
    max_tokens=512,
)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
