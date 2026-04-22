import time
import json
import os

from src.retrieval.vector_search import VectorSearch
from src.retrieval.hybrid_search import HybridSearch
from src.reranker.reranker import Reranker
from src.llm.generator import LLMGenerator
from src.memory.chat_memory import ChatMemory

CHUNKS_PATH = "data/processed/chunks.json"


class RAGPipeline:
    def __init__(self):
        print("[PIPELINE] Initializing...")

        self.search = VectorSearch()
        self.hybrid = HybridSearch()
        self.reranker = Reranker()
        self.llm = LLMGenerator()
        self.memory = ChatMemory()

        print("[PIPELINE] Ready")

    # =========================================
    # MAIN RUN FUNCTION
    # =========================================
    def run(self, query: str, user_id: str = "default"):
        start_time = time.time()

        # =========================================
        # CHAT MEMORY CONTEXT
        # =========================================
        chat_context = self.memory.get_context(user_id)

        # =========================================
        # SYSTEM QUERY: LIST FILES / METADATA
        # =========================================
        if "all files" in query.lower() or "metadata" in query.lower():
            if not os.path.exists(CHUNKS_PATH):
                return {
                    "answer": "No documents available.",
                    "sources": [],
                    "retrieved_chunks": [],
                    "scores": [],
                    "reranked": [],
                    "metadata": [],
                    "latency": 0
                }

            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            files = list(set([
                c.get("metadata", {}).get("source", "unknown")
                for c in chunks
            ]))

            return {
                "answer": "Available documents:\n" + "\n".join(files),
                "sources": files,
                "retrieved_chunks": [],
                "scores": [],
                "reranked": [],
                "metadata": [],
                "latency": 0
            }

        # =========================================
        # SAFETY: EMPTY INDEX
        # =========================================
        if self.search.index.ntotal == 0:
            return {
                "answer": "No data indexed yet. Please upload documents.",
                "sources": [],
                "retrieved_chunks": [],
                "scores": [],
                "reranked": [],
                "metadata": [],
                "latency": 0
            }

        # =========================================
        # HYBRID SEARCH (MAIN RETRIEVAL)
        # =========================================
        print("[INFO] Hybrid search...")
        results = self.hybrid.search(query, top_k=15)

        if not results:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "retrieved_chunks": [],
                "scores": [],
                "reranked": [],
                "metadata": [],
                "latency": 0
            }

        # =========================================
        # DEBUG (VERY IMPORTANT)
        # =========================================
        print("\n[DEBUG RETRIEVED SOURCES]")
        for r in results[:10]:
            print(r.get("metadata", {}).get("source", "unknown"))
        print("------------------------\n")

        # =========================================
        # RERANKING
        # =========================================
        print("[INFO] Reranking...")
        reranked = self.reranker.rerank(query, results)

        # =========================================
        # CONTEXT LIMIT (IMPORTANT)
        # =========================================
        MAX_CONTEXT = 3
        top_results = reranked[:MAX_CONTEXT]

        # =========================================
        # BUILD CONTEXT
        # =========================================
        context = "\n\n".join([r["text"] for r in top_results])

        # =========================================
        # LLM GENERATION
        # =========================================
        print("[INFO] Generating answer...")
        final_prompt = chat_context + "\n\n" + context
        answer = self.llm.generate(query, final_prompt)

        # =========================================
        # UPDATE MEMORY
        # =========================================
        self.memory.add_message(user_id, query, answer)

        # =========================================
        # SOURCES
        # =========================================
        sources = list(set([
            r.get("metadata", {}).get("source", "unknown")
            for r in top_results
        ]))

        # =========================================
        # LATENCY
        # =========================================
        latency = round(time.time() - start_time, 3)

        # =========================================
        # FINAL OUTPUT
        # =========================================
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": [r["text"] for r in results],
            "scores": [float(r.get("score", 0)) for r in results],
            "reranked": top_results,
            "metadata": [r.get("metadata", {}) for r in top_results],
            "latency": latency
        }