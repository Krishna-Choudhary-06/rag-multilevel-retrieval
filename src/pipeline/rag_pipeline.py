import time

from src.retrieval.hybrid_search import HybridSearch
from src.reranker.reranker import Reranker
from src.llm.generator import LLMGenerator
from src.memory.chat_memory import ChatMemory


class RAGPipeline:
    def __init__(self):
        print("[PIPELINE] Initializing...")
        self.hybrid = HybridSearch()
        self.reranker = Reranker()
        self.llm = LLMGenerator()
        self.memory = ChatMemory()
        print("[PIPELINE] Ready")

    # =========================
    # METADATA LAYER
    # =========================
    def metadata_layer(self, query):
        filters = {}
        q = query.lower()

        if "dbms" in q:
            filters["doc_id"] = "DBMS Notes.pdf"

        if "hackathon" in q:
            filters["doc_id"] = "SCSE-AIML-HACKATHON-PS.pdf"

        if "pdf" in q:
            filters["file_type"] = "pdf"

        return filters

    # =========================
    # DEEP RESEARCH LAYER
    # =========================
    def deep_research(self, query, context):
        prompt = f"""
Rewrite the query to improve retrieval.

Original: {query}

Context:
{context}

Better query:
"""
        try:
            return self.llm.generate(query, prompt)
        except:
            return query

    # =========================
    # MAIN RUN
    # =========================
    def run(self, query, user_id="default", filters=None, mode="Auto"):
        start_time = time.time()

        chat_context = self.memory.get_context(user_id)

        # -------------------------
        # METADATA FILTER
        # -------------------------
        auto_filters = self.metadata_layer(query)

        if filters:
            auto_filters.update(filters)

        print(f"[MODE] {mode}")
        print(f"[FILTERS] {auto_filters}")

        def format_chunk(r):
            meta = r.get("metadata", {})
            return {
                "text": r.get("text", ""),
                "metadata": meta,
                "score": r.get("score", 0.0)
            }
        
        def get_source_list(chunks):
            sources = set()
            for i, c in enumerate(chunks):
                meta = c.get("metadata", {})
                doc_id = meta.get("doc_id") or meta.get("source") or f"untracked_doc_{i}"
                sources.add(doc_id)
            return list(sources)

        # =========================
        # MODE 1: METADATA ONLY
        # =========================
        if mode == "Metadata Only":
            results = self.hybrid.search(query, top_k=10, filters=auto_filters)
            mapped_results = [format_chunk(r) for r in results]

            return {
                "answer": "Metadata filtered results (No LLM Generation)",
                "sources": get_source_list(mapped_results),
                "retrieved_chunks": mapped_results,
                "reranked": [],
                "scores": [r["score"] for r in mapped_results],
                "latency": round(time.time() - start_time, 3)
            }

        # =========================
        # MODE 2: SEMANTIC ONLY
        # =========================
        if mode == "Semantic Only":
            results = self.hybrid.search(query, top_k=15, filters=auto_filters)

            if not results:
                return {"answer": "No results found", "sources": [], "retrieved_chunks": [], "reranked": [], "latency": 0}

            reranked = self.reranker.rerank(query, results)
            top_results = reranked[:3]

            context = "\n\n".join([r["text"] for r in top_results])
            answer = self.llm.generate(query, context)

            return {
                "answer": answer,
                "sources": get_source_list(top_results),
                "retrieved_chunks": [format_chunk(r) for r in results],
                "reranked": [format_chunk(r) for r in reranked],
                "scores": [r.get("score") for r in top_results],
                "latency": round(time.time() - start_time, 3)
            }

        # =========================
        # MODE 3: DEEP (FULL)
        # =========================
        results = self.hybrid.search(query, top_k=15, filters=auto_filters)

        if not results:
            return {"answer": "No results found", "sources": [], "retrieved_chunks": [], "reranked": [], "latency": 0}

        preview = "\n".join([r["text"] for r in results[:3]])
        expanded_query = self.deep_research(query, preview)

        deep_results = self.hybrid.search(expanded_query, top_k=15, filters=auto_filters)

        combined = results + deep_results

        # remove duplicates
        seen = set()
        unique = []
        for r in combined:
            key = r["text"][:100]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        reranked = self.reranker.rerank(query, unique)
        top_results = reranked[:6]

        context = "\n\n".join([r["text"] for r in top_results])

        answer = self.llm.generate(query, context)

        self.memory.add_message(user_id, query, answer)

        return {
            "answer": answer,
            "sources": get_source_list(top_results),
            "retrieved_chunks": [format_chunk(r) for r in unique],
            "reranked": [format_chunk(r) for r in reranked],
            "scores": [r.get("score") for r in top_results],
            "latency": round(time.time() - start_time, 3)
        }