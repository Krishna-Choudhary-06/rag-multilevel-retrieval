import json
from rank_bm25 import BM25Okapi


METADATA_PATH = "embeddings/metadata_store/metadata.json"


class HybridSearch:
    def __init__(self):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.texts = [item["text"] for item in self.metadata]
        self.tokenized_texts = [text.lower().split() for text in self.texts]

        self.bm25 = BM25Okapi(self.tokenized_texts)

    def bm25_search(self, query, top_k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return ranked_indices