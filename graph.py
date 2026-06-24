from langgraph.graph import StateGraph, START, END
from state import State
import nodes

# Initialize the state graph
g = StateGraph(State)

# Add all nodes
g.add_node("decide_retrieval", nodes.decide_retrieval)
g.add_node("is_relevant", nodes.is_relevant)
g.add_node("generate_from_context", nodes.generate_from_context)
g.add_node("is_sup", nodes.is_sup)
g.add_node("accept_revised_answer", nodes.accept_revised_answer)
g.add_node("revise_answer", nodes.revise_answer)
g.add_node("no_relevant_docs", nodes.no_relevant_docs)
g.add_node("generate_direct", nodes.generate_direct)
g.add_node("retrieve", nodes.rertrieve)
g.add_node("is_useful", nodes.is_useful)

# Define edges
g.add_edge(START, "decide_retrieval")

# Conditional edges after retrieval decision
g.add_conditional_edges(
    "decide_retrieval",
    nodes.route_after_decision,
    {
        "generate_direct": "generate_direct",
        "retrieve": "retrieve"
    }
)

# Flow from retrieve to relevance check
g.add_edge("retrieve", "is_relevant")

# Conditional edges after relevance check
g.add_conditional_edges(
    "is_relevant",
    nodes.route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_relevant_docs": "no_relevant_docs"
    }
)

# Flow from RAG generation to support validation
g.add_edge("generate_from_context", "is_sup")

# Conditional edges after support validation
g.add_conditional_edges(
    "is_sup",
    nodes.route_after_issup,
    {
        "revise_answer": "revise_answer",
        "accept_revised_answer": "is_useful"
    }
)

# Loop back from revise_answer to support check
g.add_edge("revise_answer", "is_sup")

# Conditional edges after usefulness validation
g.add_conditional_edges(
    "is_useful",
    nodes.route_after_is_useful,
    {
        "finalize": "accept_revised_answer",
        "no_answer_found": "no_relevant_docs"
    }
)

# Final exit edges to END
g.add_edge("generate_direct", END)
g.add_edge("accept_revised_answer", END)
g.add_edge("no_relevant_docs", END)

# Compile graph
app = g.compile()
