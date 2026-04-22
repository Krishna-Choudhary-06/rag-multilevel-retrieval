from src.retrieval.metadata_filter import MetadataFilter

mf = MetadataFilter()

results = mf.filter({"doc_id": "rag_intro.txt"})

print(f"Filtered results: {len(results)}")

for r in results:
    print(r["text"][:1000])
