def chunk_documents(docs, chunk_size=500, overlap=50):
    chunks = []

    for doc in docs:
        text = doc["text"]
        metadata = doc["metadata"]

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": chunk_id
                }
            })

            start += chunk_size - overlap
            chunk_id += 1

    return chunks