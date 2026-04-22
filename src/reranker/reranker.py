import torch
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"


class Reranker:
    def __init__(self):
        print(f"[RERANKER] Using device: {device}")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    def rerank(self, query, results):
        if not results:
            return []

        pairs = [(query, r["text"]) for r in results]

        scores = self.model.predict(pairs, batch_size=16)  # 🔥 SPEED BOOST

        for r, score in zip(results, scores):
            r["score"] = float(score)

        return sorted(results, key=lambda x: x["score"], reverse=True)
