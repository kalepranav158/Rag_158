from pydantic import BaseModel, Field
from typing import Literal, TypedDict,List
from langchain_core.documents import Document


# -----------------------------
#       GRAPH STATE
#-----------------------------
class State(TypedDict):
    question :str
    need_retrieval:bool
    docs: List[Document]
    relevant_docs: List[Document]
    context:str
    
    issup:Literal["fully_supported","partially_supported","not_supported"]
    evidence:List[str]


    answer:str
    issue:Literal["useful","not_useful"]

    retries:int

