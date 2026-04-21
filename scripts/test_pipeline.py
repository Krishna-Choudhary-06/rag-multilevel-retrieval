from src.pipeline.rag_pipeline import RAGPipeline

rag = RAGPipeline()

query = "Explain retrieval augmented generation"

response = rag.run(query)

print("\n=== FINAL ANSWER ===\n")
print(response["answer"])

print("\n=== SOURCES ===")
for s in response["sources"]:
    print("-", s)