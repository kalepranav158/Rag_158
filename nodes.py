from typing import List, Literal
from langsmith import traceable
from langchain_core.documents import Document
from config import judge_llm, generator_llm 
from reranker import reranker
from state import State
from retriever import hybrid_retrieve
from models import RetriveDecision, RelevanceDecision, IsSUPDecision, IsUSEDecision
from prompts import (
    decide_retrieval_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
    direct_generation_prompt,
    usefulness_prompt,
    rewrite_query_prompt
)
import state

# Initialize retriever
retriever = hybrid_retrieve


# LLM bindings with structured output
should_retrieve_llm = judge_llm.with_structured_output(RetriveDecision)

relevance_llm = judge_llm.with_structured_output(RelevanceDecision)
issup_llm = judge_llm.with_structured_output(IsSUPDecision)
issue_llm = judge_llm.with_structured_output(IsUSEDecision)

# -----------------------------
#       GRAPH NODES
# -----------------------------

@traceable(name="decide_retrieval")
def decide_retrieval(state: State):
    print("DECIDE RETRIEVAL NODE CALLED")
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
    print("IS RELEVANT NODE CALLED")
    relevant_docs: List[Document] = []


    state["reranked_docs"] = reranker.rerank(
    query=state["question"],
    docs=state["docs"],
    top_k=5
)
    
    for i, doc in enumerate(state["reranked_docs"], start=1):
        decision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            )
        )

        print("Decision:", decision.model_dump())

        if decision.is_relevant:
            relevant_docs.append(doc)
            print("✅ Added to relevant_docs")
        else:
            print("❌ Rejected")

 
    return {
        "relevant_docs": relevant_docs
    }

@traceable(name="generate_from_context")
def generate_from_context(state: State):
    print("GENERATE FROM CONTEXT NODE CALLED")
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No relevant Document found", "context": ""}
    
    out = generator_llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )
    return {"answer": out.content, "context": context}

@traceable(name="is_sup")
def is_sup(state: State):
    print("IS SUP NODE CALLED")
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
    print("ACCEPT REVISED ANSWER NODE CALLED")
    return {
        "answer": state["answer"],
        "context": state.get("context", "")
    }

@traceable(name="revise_answer")
def revise_answer(state: State):
    print("REVISE ANSWER NODE CALLED")
    revised_answer = generator_llm.invoke(
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
    print("In NO RELEVANT DOCS NODE")
    return {"answer": "No relevant Document found", "context": ""}

@traceable(name="generate_direct")
def generate_direct(state: State):
    print("GENERATE DIRECT_NODE CALLED")
    out = generator_llm.invoke(
        direct_generation_prompt.format_prompt(question=state["question"])
    )
    return {"answer": out.content}

@traceable(name="retrieve")
def retrieve(state: State):
    print("RETRIEVE NODE CALLED")
    query = state.get("rewritten_question")

    if not query:
        query = state["question"]


    return {
        "docs": retriever(query)
    }

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



@traceable(name="rewrite_question")
def rewrite_question(state: State):

    rewritten = judge_llm.invoke(
        rewrite_query_prompt.format_messages(
            question=state["question"]
        )
    )

    print("\nQUERY REWRITE")
    print("Original :", state["question"])
    print("Rewritten:", rewritten.content)

    return {
    "rewritten_question": rewritten.content,
    "rewrite_attempts": state.get("rewrite_attempts", 0) + 1,
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
def route_after_issup(
    state: State,
) -> Literal["revise_answer", "accept_revised_answer"]:

    MAX_RETRIES = 2

    # Stop revising after 2 attempts
    if state.get("retries", 0) >= MAX_RETRIES:
        print(f"Maximum revision attempts ({MAX_RETRIES}) reached.")
        return "accept_revised_answer"

    # Answer is fully grounded
    if state.get("issup") == "fully_supported":
        return "accept_revised_answer"

    # Otherwise, revise and check again
    return "revise_answer"

@traceable(name="route_after_is_useful")
def route_after_is_useful(state: State) -> Literal["no_answer_found", "finalize"]:
    if state.get("issue") == "useful":
        return "finalize"
    else:
        return "no_answer_found"
