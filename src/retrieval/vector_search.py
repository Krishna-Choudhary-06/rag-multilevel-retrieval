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
    def search(self, query, top_k=15):
        if self.index.ntotal == 0:
            print("[FAISS] Empty index")
            return []

        query_embedding = self.model.encode(query)
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append(
                    {
                        "score": float(score),
                        "text": self.metadata[idx]["text"],
                        "metadata": self.metadata[idx]["metadata"],
                    }
                )

        return results[:top_k]
