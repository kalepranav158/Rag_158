from langchain_core.prompts import ChatPromptTemplate

# Original Basic RAG Prompt
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI assistant.

Your task is to answer the user's question ONLY using the retrieved context.

Instructions:
1. Use the provided context as the primary source of truth.
2. If the answer is fully supported by the context, answer clearly and concisely.
3. If the context partially answers the question, explain what is available and what is missing.
4. If the answer cannot be found in the context, respond:

   "I don't know based on the provided documents."

5. Do not invent facts.
6. Do not use outside knowledge.
7. Cite important facts using information from the context.
"""),
    ("human", """ Question: {question}
Context: {context}
""")
])

# RAG Graph Prompts

decide_retrieval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that decides whether to retrieve external documents to answer the question or not.\n"
        "Return JSON that matches strictly this schema:\n"
        "Analyze the question if it mention the docuement then give True, else give False.\n"
        "{{'should_retrieve':boolean}}\n"
        "Guidelines:\n"
        "- Should retrieve True if answering requires specific facts, citations, or info that is not likely in the model, for general knowledge.\n"
        "- Should retrieve False if the question can be answered based on general knowledge or reasoning that is present in the model parameters. Strictly give false if the model can answer a general question.\n"
        "- If unsure and the question is related to documents, choose True."
    ),
    ("human", "Question: {question}"),
])

is_relevant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are judging document relevance.\n"
        "Return JSON that matches this schema:\n"
        "{{'is_relevant':boolean}}\n\n"
        "A document is relevant if it contains the information useful for answering the question."
    ),
    ("human", "Question:\n{question}\n\nDocument:\n{document}"),
])

rag_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business RAG Assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the context does not contain enough information, say: No relevant Documents found.\n"
        "Do not use outside knowledge."
    ),
    (
        "human",
        "Question: {question}\n"
        "Context: {context}"
    ),
])

issup_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are verifying whether the ANSWER is supported by the CONTEXT.\n\n"
        "Return JSON with keys: issup, evidence.\n\n"
        "issup must be one of: fully_supported, partially_supported, not_supported.\n\n"
        "How to decide issup:\n\n"
        "- fully_supported:\n"
        "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce any qualitative/interpretive words that are not present in CONTEXT.\n"
        "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to, supports professional development, best-in-class, employee-first, etc.)\n\n"
        "- partially_supported:\n"
        "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative phrasing not explicitly stated in CONTEXT\n"
        "  (e.g., calling policies \"culture\", saying leave is \"generous\", or inferring outcomes like \"supports professional development\").\n\n"
        "- not_supported:\n"
        "  The key claims are not supported by CONTEXT.\n\n"
        "Rules:\n"
        "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
        "- If the answer is mostly unrelated to the question or unsupported, choose not_supported.\n"
        "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
        "- Do not use outside knowledge.\n"
    ),
    (
        "human",
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer: {answer}"
    ),
])

revise_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a STRICT reviser.\n\n"
        "You must output based on the following format:\n\n"
        "FORMAT (quote-only answer):\n\n"
        "- <direct quote from the CONTEXT>\n"
        "- <direct quote from the CONTEXT>\n\n"
        "Rules:\n"
        "- Use ONLY the CONTEXT.\n"
        "- DO NOT add any new words besides bullet dashes and the quotes themselves.\n"
        "- DO NOT explain anything.\n"
        "- DO NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
    ),
    (
        "human",
        "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"
    ),
])

direct_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions based on the provided context."
    ),
    ("human", "{question}"),
])

usefulness_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
        "Goal:\n"
        "- Decide if the answer actually addresses what the user asked.\n\n"
        "Return JSON with keys: issue, reason.\n"
        "issue must be one of: useful, not_useful.\n\n"
        "Rules:\n"
        "If user ask for theme or summary of main topic analyze the answer and decide if it is useful or not.\n"
        "- useful: The answer directly answers the question or provides the requested specific info.\n"
        "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
        "- Do NOT use outside knowledge.\n"
        "- Do NOT re-check grounding (ISSUP already did that). Only check: 'Did we answer the question?'\n"
        "- Keep reason to 1 short line."
    ),
    (
        "human",
        "Question:\n{question}\n\nAnswer:\n{answer}"
    )
])