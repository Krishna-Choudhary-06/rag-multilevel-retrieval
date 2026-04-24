import os
import json
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.embeddings.update_index import update_faiss_index

DATA_PATH = "data/uploaded"
CHUNKS_PATH = "data/processed/chunks.json"


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path):
    import fitz  # PyMuPDF

    text = ""
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        text += page.get_text()

    return text


def ingest_uploaded():
    print("[INGEST] Starting ingestion...")

    os.makedirs("data/processed", exist_ok=True)

    files = os.listdir(DATA_PATH)
    print(f"[INGEST] Files found: {files}")

    # -------------------------
    # LOAD EXISTING CHUNKS
    # -------------------------
    if os.path.exists(CHUNKS_PATH):
        try:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        except:
            existing_chunks = []
    else:
        existing_chunks = []

    existing_texts = set([c["text"] for c in existing_chunks])

    new_chunks = []

    # -------------------------
    # PROCESS FILES
    # -------------------------
    for file in files:
        file_path = os.path.join(DATA_PATH, file)

        # skip empty files
        if os.path.getsize(file_path) == 0:
            print(f"[SKIP] Empty file: {file}")
            continue

        try:
            if file.endswith(".pdf"):
                text = load_pdf(file_path)
            else:
                text = load_text(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")
            continue

        if not text.strip():
            print(f"[SKIP] No text extracted: {file}")
            continue

        docs = [{
            "text": text,
            "metadata": {
                "doc_id": file,
                "file_type": file.split(".")[-1]
            }
        }]

        chunks = chunk_documents(docs)

        print(f"[CHUNKER] {file} → {len(chunks)} chunks")

        # -------------------------
        # REMOVE DUPLICATES
        # -------------------------
        for c in chunks:
            if c["text"] not in existing_texts:
                new_chunks.append(c)
                existing_texts.add(c["text"])

    # -------------------------
    # MERGE + SAVE
    # -------------------------
    all_chunks = existing_chunks + new_chunks

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[INGEST] New chunks added: {len(new_chunks)}")
    print(f"[INGEST] Total chunks: {len(all_chunks)}")

    # -------------------------
    # UPDATE FAISS
    # -------------------------
    if new_chunks:
        update_faiss_index(new_chunks)
    else:
        print("[INGEST] No new chunks to embed")

    return len(new_chunks)