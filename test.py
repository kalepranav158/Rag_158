from dotenv import load_dotenv
from chain_main import rag_answer
load_dotenv()


ans = rag_answer("What is the main topic of the document?")
print(ans)


