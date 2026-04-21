import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "embeddings/faiss_index/index.faiss"
METADATA_PATH = "embeddings/metadata_store/metadata.json"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(chunks):
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "metadata": c["metadata"]
        }
        for c in chunks
    ]

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    print("[INFO] Loading chunks...")
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    print(f"[INFO] Loaded {len(texts)} chunks")

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-en")

    print("[INFO] Generating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("[INFO] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    print("[INFO] Saving metadata...")
    save_metadata(chunks)

    print("[SUCCESS] Index built and saved.")


if __name__ == "__main__":
    main()