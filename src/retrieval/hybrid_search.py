import json
import os

from src.retrieval.vector_search import VectorSearch
from src.retrieval.bm25_search import BM25Search

CHUNKS_PATH = "data/processed/chunks.json"


class HybridSearch:
    def __init__(self):
        print("[HYBRID] Initializing...")

        self.vector = VectorSearch()
        self.bm25 = BM25Search()
        self.chunks = []

    # =========================
    # 🔥 ALWAYS LOAD LATEST DATA
    # =========================
    def _reload_chunks(self):
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            print("[HYBRID] No chunks found")
            self.chunks = []

    # =========================
    # MAIN SEARCH
    # =========================
    def search(self, query, top_k=10):
        print("[HYBRID] Running hybrid search...")

        # 🔥 Reload latest chunks every query
        self._reload_chunks()

        if not self.chunks:
            return []

        # =========================
        # VECTOR SEARCH
        # =========================
        vector_results = self.vector.search(query, top_k=top_k)

        # =========================
        # BM25 SEARCH (RELOAD INSIDE)
        # =========================
        bm25_results = self.bm25.search(query, top_k=top_k)

        # =========================
        # MERGE RESULTS
        # =========================
        combined = {}

        # 🔹 Vector weight (semantic)
        VECTOR_WEIGHT = 0.7

        for r in vector_results:
            key = r["text"]

            combined[key] = {
                "text": r["text"],
                "score": float(r.get("score", 0)) * VECTOR_WEIGHT,
                "metadata": r.get("metadata", {}),
            }

        # 🔹 BM25 weight (keyword)
        BM25_WEIGHT = 0.3

        for r in bm25_results:
            key = r["text"]

            if key in combined:
                combined[key]["score"] += float(r.get("score", 0)) * BM25_WEIGHT
            else:
                combined[key] = {
                    "text": r["text"],
                    "score": float(r.get("score", 0)) * BM25_WEIGHT,
                    "metadata": r.get("metadata", {}),
                }

        # =========================
        # SORT RESULTS
        # =========================
        final_results = sorted(
            combined.values(), key=lambda x: x["score"], reverse=True
        )

        print(f"[HYBRID] Total merged results: {len(final_results)}")

        # =========================
        # 🔍 DEBUG (CRITICAL FOR YOU)
        # =========================
        print("\n[HYBRID DEBUG SOURCES]")
        for r in final_results[:5]:
            print(r.get("metadata", {}).get("source", "unknown"))
        print("------------------------\n")

        return final_results[:top_k]
