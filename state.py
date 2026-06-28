from pydantic import BaseModel, Field
from typing import Literal, TypedDict,List
from langchain_core.documents import Document


# -----------------------------
#       GRAPH STATE
#-----------------------------
from typing import TypedDict, List
from langchain_core.documents import Document

class State(TypedDict, total=False):

    question: str
    rewritten_question: str

    docs: List[Document]
    relevant_docs: List[Document]
    reranked_docs: List[Document]

    answer: str
    context: str

    retries: int
    rewrite_attempts: int

    need_retrieval: bool

    issup: str
    issue: str

    evidence: list
    reason: str