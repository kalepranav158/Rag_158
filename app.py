import sys
from dotenv import load_dotenv
from langsmith import traceable
from graph import app

# Load environment variables (e.g. LangSmith keys, etc)
load_dotenv()

@traceable(name="run_self_rag")
def run_self_rag(question: str):
    print(f"Running Self-RAG for query: '{question}'\n")
    
    # Initialize the state
    initial_state = {
        "question": question,
        "retries": 0,
        "relevant_docs": [],
        "docs": [],
        "rewrite_attempts": 0,
        "answer": ""
    }
    
    # Execute the graph
    result = app.invoke(initial_state)
    
    print("\n--- Final Results ---")
    print(f"Question: {result.get('question')}")
    print(f"Answer: {result.get('answer')}")
    print(f"Need Retrieval: {result.get('need_retrieval')}")
   
    print(f"Retrieved Documents Count: {len(result.get('docs', []))}")
   
    

if __name__ == "__main__":
    query = "What is this paper about?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    run_self_rag(query)
