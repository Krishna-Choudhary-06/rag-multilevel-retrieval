import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "src/embeddings/faiss_index/index.faiss"
META_PATH = "src/embeddings/faiss_index/metadata.json"


class VectorSearch:
    def __init__(self):
        print("[VECTOR] Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-small-en")

        # =========================
        # LOAD OR CREATE INDEX
        # =========================
        if os.path.exists(INDEX_PATH):
            print("[FAISS] Loading existing index...")
            self.index = faiss.read_index(INDEX_PATH)
        else:
            print("[FAISS] No index found. Creating default FLAT index...")
            dim = 384
            self.index = faiss.IndexFlatL2(dim)

        # =========================
        # SAFE nprobe (ONLY FOR IVF)
        # =========================
        if isinstance(self.index, faiss.IndexIVF):
            print("[FAISS] IVF index detected → setting nprobe")
            self.index.nprobe = 10
        else:
            print("[FAISS] Flat index detected → skipping nprobe")

        # =========================
        # LOAD METADATA
        # =========================
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    # =========================
    # SEARCH
    # =========================
    def search(self, query, top_k=5, filters=None):
        # -------------------------
        # SAFETY CHECK
        # -------------------------
        if self.index is None or self.index.ntotal == 0:
            return []

        # -------------------------
        # ENCODE QUERY
        # -------------------------
        query_vec = self.model.encode([query])
        query_vec = query_vec.astype("float32")

        # -------------------------
        # SEARCH
        # -------------------------
        scores, indices = self.index.search(query_vec, top_k * 2)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.metadata):
                continue

            item = self.metadata[idx]

            # -------------------------
            # APPLY FILTERS
            # -------------------------
            if filters:
                skip = False
                for k, v in filters.items():
                    if item["metadata"].get(k) != v:
                        skip = True
                        break

                if skip:
                    continue

            results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "score": float(score)
            })

        return results[:top_k]