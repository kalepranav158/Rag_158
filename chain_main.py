from langsmith import traceable

from retriever import hybrid_retrieve
from prompts import generate_prompt
from config import llm


@traceable(name="basic_rag")
def rag_answer(question: str):

    docs = hybrid_retrieve(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    chain = generate_prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    return response.content