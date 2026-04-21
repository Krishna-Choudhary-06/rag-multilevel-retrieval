from src.ingestion.chunker import chunk_documents

documents = [
    {
        "text": "This is a long document. " * 100,
        "metadata": {
            "doc_id": "doc1",
            "source": "test"
        }
    }
]

chunks = chunk_documents(documents, chunk_size=200, overlap=50)

print(f"Total chunks: {len(chunks)}")
print(chunks[0])