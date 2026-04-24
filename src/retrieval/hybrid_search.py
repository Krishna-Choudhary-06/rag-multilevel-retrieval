from src.retrieval.vector_search import VectorSearch
from src.retrieval.bm25_search import BM25Search


class HybridSearch:
    def __init__(self):
        print("[HYBRID] Initializing...")

        self.vector = VectorSearch()
        self.bm25 = BM25Search()

        # weights (can tune later)
        self.vector_weight = 0.6
        self.bm25_weight = 0.4

    # =========================
    # NORMALIZE SCORES
    # =========================
    def normalize(self, scores):
        if not scores:
            return scores

        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [1.0 for _ in scores]

        return [(s - min_s) / (max_s - min_s) for s in scores]

    # =========================
    # MAIN SEARCH
    # =========================
    def search(self, query, top_k=10, filters=None):
        print("[HYBRID] Running hybrid search...")

        # -------------------------
        # VECTOR SEARCH
        # -------------------------
        vector_results = self.vector.search(query, top_k=top_k, filters=filters)

        # -------------------------
        # BM25 SEARCH
        # -------------------------
        bm25_results = self.bm25.search(query, top_k=top_k, filters=filters)

        # -------------------------
        # NORMALIZE SCORES
        # -------------------------
        vector_scores = [r["score"] for r in vector_results]
        bm25_scores = [r["score"] for r in bm25_results]

        vector_scores = self.normalize(vector_scores)
        bm25_scores = self.normalize(bm25_scores)

        # assign normalized scores back
        for i, r in enumerate(vector_results):
            r["score"] = vector_scores[i] * self.vector_weight

        for i, r in enumerate(bm25_results):
            r["score"] = bm25_scores[i] * self.bm25_weight

        # -------------------------
        # MERGE RESULTS
        # -------------------------
        combined = vector_results + bm25_results

        # -------------------------
        # REMOVE DUPLICATES (SMART)
        # -------------------------
        seen = {}
        unique = []

        for r in combined:
            key = r["text"][:150]  # fingerprint

            if key not in seen:
                seen[key] = r
                unique.append(r)
            else:
                # keep higher score
                if r["score"] > seen[key]["score"]:
                    seen[key] = r

        # rebuild unique list from dict
        unique = list(seen.values())

        # -------------------------
        # SORT BY FINAL SCORE
        # -------------------------
        unique = sorted(unique, key=lambda x: x["score"], reverse=True)

        print(f"[HYBRID] Final results: {len(unique)}")

        return unique[:top_k]