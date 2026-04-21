import faiss
import json
import numpy as np

from sentence_transformers import SentenceTransformer


INDEX_PATH = "embeddings/faiss_index/index.faiss"
METADATA_PATH = "embeddings/metadata_store/metadata.json"


class VectorSearch:
    def __init__(self, model_name="BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _apply_metadata_filter(self, filters):
        if not filters:
            return set(range(len(self.metadata)))

        valid_indices = set()

        for idx, item in enumerate(self.metadata):
            match = True
            for key, value in filters.items():
                if item["metadata"].get(key) != value:
                    match = False
                    break

            if match:
                valid_indices.add(idx)

        return valid_indices

    def search(self, query: str, top_k: int = 5, filters: dict = None):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding, top_k * 5)

        valid_indices = self._apply_metadata_filter(filters)

        results = []

        for i, idx in enumerate(indices[0]):
            if idx in valid_indices:
                results.append({
                    "score": float(scores[0][i]),
                    "text": self.metadata[idx]["text"],
                    "metadata": self.metadata[idx]["metadata"]
                })

            if len(results) >= top_k:
                break

        return results