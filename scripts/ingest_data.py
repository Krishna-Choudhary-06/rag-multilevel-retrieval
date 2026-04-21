import json
import os

from src.ingestion.chunker import chunk_documents


RAW_DATA_PATH = "data/raw/"
OUTPUT_PATH = "data/processed/chunks.json"


def load_raw_documents():
    """
    Loads all .txt files from raw data folder
    """

    documents = []

    for filename in os.listdir(RAW_DATA_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(RAW_DATA_PATH, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "text": text,
                "metadata": {
                    "doc_id": filename,
                    "source": "local"
                }
            })

    return documents


def save_chunks(chunks):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)


def main():
    print("[INFO] Loading raw documents...")
    documents = load_raw_documents()
    print(f"[INFO] Loaded {len(documents)} documents")

    print("[INFO] Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=500, overlap=80)
    print(f"[INFO] Created {len(chunks)} chunks")

    print("[INFO] Saving chunks...")
    save_chunks(chunks)

    print("[SUCCESS] Ingestion complete → data/processed/chunks.json")


if __name__ == "__main__":
    main()