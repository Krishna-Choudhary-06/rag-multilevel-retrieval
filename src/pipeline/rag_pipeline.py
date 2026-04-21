from src.retrieval.vector_search import VectorSearch
from src.retrieval.hybrid_search import HybridSearch
from src.reranker.rerank import Reranker
from src.llm.generator import LLMGenerator
from src.utils.cache import QueryCache


class RAGPipeline:
    def __init__(self):
        self.search = VectorSearch()
        self.hybrid = HybridSearch()
        self.reranker = Reranker()
        self.llm = LLMGenerator()
        self.cache = QueryCache()

    def build_context(self, docs, max_chunks=2):
        return "\n\n".join([d["text"] for d in docs[:max_chunks]])

    def run(self, query, top_k=3, filters=None):

        # CACHE KEY NORMALIZATION
        cache_key = query.strip().lower()

        cached = self.cache.get(cache_key)
        if cached:
            print("[CACHE HIT]")
            return cached

        print("[CACHE MISS] Running hybrid pipeline...")

        # STEP 1 — VECTOR SEARCH
        print("[INFO] Vector search...")
        vector_results = self.search.search(query, top_k=10, filters=filters)

        # STEP 2 — BM25 SEARCH
        print("[INFO] BM25 search...")
        bm25_indices = self.hybrid.bm25_search(query, top_k=5)

        # STEP 3 — MERGE RESULTS
        print("[INFO] Merging results...")
        combined = {}

        # Add vector results
        for r in vector_results:
            key = r["metadata"]["doc_id"] + "_" + r["text"][:30]
            combined[key] = r

        # Add BM25 results
        for idx in bm25_indices:
            item = self.hybrid.metadata[idx]
            key = item["metadata"]["doc_id"] + "_" + item["text"][:30]

            if key not in combined:
                combined[key] = {
                    "score": 0,
                    "text": item["text"],
                    "metadata": item["metadata"]
                }

        final_candidates = list(combined.values())

        # STEP 4 — RERANK
        print("[INFO] Reranking...")
        reranked = self.reranker.rerank(query, final_candidates, top_k=top_k)

        # STEP 5 — BUILD CONTEXT
        context = self.build_context(reranked)

        # STEP 6 — LLM GENERATION
        print("[INFO] Generating answer...")
        answer = self.llm.generate(query, context)

        output = {
            "query": query,
            "answer": answer.strip(),
            "sources": [d["metadata"]["doc_id"] for d in reranked]
        }

        # STEP 7 — STORE CACHE
        self.cache.set(cache_key, output)

        return output