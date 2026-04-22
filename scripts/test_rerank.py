from src.retrieval.vector_search import VectorSearch
from src.reranker.rerank import Reranker

search_engine = VectorSearch()
reranker = Reranker()

query = "What is retrieval augmented generation?"

# Step 1: retrieve more candidates
results = search_engine.search(query, top_k=5)

print("\n--- BEFORE RERANK ---")
for r in results:
    print(r["score"], r["metadata"]["doc_id"])


# Step 2: rerank
reranked = reranker.rerank(query, results, top_k=3)

print("\n--- AFTER RERANK ---")
for r in reranked:
    print(r["rerank_score"], r["metadata"]["doc_id"])
