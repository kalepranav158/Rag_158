from typing import List, Literal
from langsmith import traceable
from langchain_core.documents import Document
from config import llm
from state import State
from retriever import get_faiss_retriever
from models import RetriveDecision, RelevanceDecision, IsSUPDecision, IsUSEDecision
from prompts import (
    decide_retrieval_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
    direct_generation_prompt,
    usefulness_prompt
)

# Initialize retriever
retriever = get_faiss_retriever()

# LLM bindings with structured output
should_retrieve_llm = llm.with_structured_output(RetriveDecision)
relevance_llm = llm.with_structured_output(RelevanceDecision)
issup_llm = llm.with_structured_output(IsSUPDecision)
issue_llm = llm.with_structured_output(IsUSEDecision)

# -----------------------------
#       GRAPH NODES
# -----------------------------

@traceable(name="decide_retrieval")
def decide_retrieval(state: State):
    decision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_prompt(
            question=state["question"]
        )
    )
    print("DECISION =", decision)
    print("TYPE =", type(decision))
    print("BOOL =", decision.should_retrieve)

    return {
        "need_retrieval": decision.should_retrieve
    }

@traceable(name="is_relevant")
def is_relevant(state: State):
    relevant_docs: List[Document] = []
    for doc in state["docs"]:
        decision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}

@traceable(name="generate_from_context")
def generate_from_context(state: State):
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No relevant Document found", "context": ""}
    
    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )
    return {"answer": out.content, "context": context}

@traceable(name="is_sup")
def is_sup(state: State):
    decision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            context=state.get("context", ""),
            answer=state.get("answer", "")
        )
    )

    return {
        "issup": decision.issup,
        "evidence": decision.evidence
    }

@traceable(name="accept_revised_answer")
def accept_revised_answer(state: State):
    return {
        "answer": state["answer"],
        "context": state.get("context", "")
    }

@traceable(name="revise_answer")
def revise_answer(state: State):
    revised_answer = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    )
    return {
        "answer": revised_answer.content,
        "retries": state.get("retries", 0) + 1
    }

@traceable(name="no_relevant_docs")
def no_relevant_docs(state: State):
    return {"answer": "No relevant Document found", "context": ""}

@traceable(name="generate_direct")
def generate_direct(state: State):
    print("GENERATE DIRECT CALLED")
    out = llm.invoke(
        direct_generation_prompt.format_prompt(question=state["question"])
    )
    return {"answer": out.content}

@traceable(name="rertrieve")
def rertrieve(state: State):
    return {"docs": retriever.invoke(state["question"])}

@traceable(name="is_useful")
def is_useful(state: State):
    decision = issue_llm.invoke(
        usefulness_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", "")
        )
    )

    return {
        "issue": decision.issue,
        "reason": decision.reason
    }

# -----------------------------
#       ROUTING LOGIC
# -----------------------------

@traceable(name="route_after_decision")
def route_after_decision(state: State) -> Literal["generate_direct", "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    else:
        return "generate_direct"

@traceable(name="route_after_relevance")
def route_after_relevance(state: State) -> Literal["generate_from_context", "no_relevant_docs"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    
    return "no_relevant_docs"

@traceable(name="route_after_issup")
def route_after_issup(state: State) -> Literal["revise_answer", "accept_revised_answer"]:
    # Check retry limit. If retries exceeded, route to accept_revised_answer (to let usefulness node assess it)
    if state.get("retries", 0) >= 3:
        print("Max retries reached, accepting current answer.")
        return "accept_revised_answer"
        
    if state.get("issup") == "fully_supported":
        return "accept_revised_answer"
    else:
        return "revise_answer"

@traceable(name="route_after_is_useful")
def route_after_is_useful(state: State) -> Literal["no_answer_found", "finalize"]:
    if state.get("issue") == "useful":
        return "finalize"
    else:
        return "no_answer_found"
