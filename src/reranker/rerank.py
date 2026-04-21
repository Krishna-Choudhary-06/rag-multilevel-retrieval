from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list, top_k: int = 3):
        """
        documents = list of dicts from vector search
        """

        pairs = [(query, doc["text"]) for doc in documents]

        scores = self.model.predict(pairs)

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        # sort by rerank score
        reranked = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]