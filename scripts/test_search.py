from src.retrieval.vector_search import VectorSearch

search_engine = VectorSearch()

query = "What is retrieval augmented generation?"

print("\n--- WITHOUT FILTER ---")
results = search_engine.search(query, top_k=3)

for r in results:
    print(r["metadata"]["doc_id"], r["score"])


print("\n--- WITH FILTER (rag_intro.txt) ---")
filtered_results = search_engine.search(
    query,
    top_k=3,
    filters={"doc_id": "rag_intro.txt"}
)

for r in filtered_results:
    print(r["metadata"]["doc_id"], r["score"])