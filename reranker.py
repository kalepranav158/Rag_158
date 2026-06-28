from typing import List

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langsmith import traceable


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print(f"Loading CrossEncoder: {model_name}")

        self.model = CrossEncoder(model_name)

        print("CrossEncoder Loaded Successfully")

    @traceable(name="cross_encoder_rerank")
    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 5,
    ) -> List[Document]:

        if not docs:
            return []

        # Create (query, document) pairs
        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Zip documents with scores
        ranked = list(zip(docs, scores))

        # Sort by descending score
        ranked.sort(
            key=lambda x: x[1],
            reverse=True
        )

        print("\n" + "=" * 100)
        print("CROSS ENCODER RANKING")
        print("=" * 100)

        for i, (doc, score) in enumerate(ranked, start=1):

            print(f"{i}. Score : {score:.4f}")
            print(f"Page     : {doc.metadata.get('page')}")
            print(doc.page_content[:200].replace("\n", " "))
            print("-" * 80)

        # Return Top-K Documents
        return [
            doc
            for doc, _ in ranked[:top_k]
        ]


# Singleton Instance
reranker = CrossEncoderReranker()