import os
import json

from src.ingestion.chunker import chunk_text
from src.embeddings.update_index import update_faiss_index

# Paths
UPLOAD_DIR = "data/uploaded"
CHUNKS_PATH = "data/processed/chunks.json"


# =========================
# FILE LOADERS
# =========================
def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path):
    import fitz  # PyMuPDF

    text = ""
    doc = fitz.open(file_path)

    for page in doc:
        text += page.get_text()

    return text


def load_csv(file_path):
    import pandas as pd

    df = pd.read_csv(file_path)
    return df.to_string()


# =========================
# MAIN INGEST FUNCTION
# =========================
def ingest_uploaded():
    print("[INGEST] Starting ingestion...")

    # =========================
    # LOAD EXISTING CHUNKS
    # =========================
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    existing_ids = set(c["id"] for c in existing_chunks)

    # =========================
    # LOAD FILES
    # =========================
    files = os.listdir(UPLOAD_DIR)
    print(f"[INGEST] Files found: {files}")

    new_chunks = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file)

        # =========================
        # LOAD FILE BASED ON TYPE
        # =========================
        if file.endswith(".txt"):
            text = load_txt(file_path)

        elif file.endswith(".pdf"):
            text = load_pdf(file_path)

        elif file.endswith(".csv"):
            text = load_csv(file_path)

        else:
            print(f"[INGEST] Skipping unsupported file: {file}")
            continue

        # =========================
        # CHUNK TEXT
        # =========================
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file}_{i}"

            # 🔥 SKIP DUPLICATES
            if chunk_id in existing_ids:
                continue

            new_chunks.append(
                {"id": chunk_id, "text": chunk, "metadata": {"source": file}}
            )

    # =========================
    # MERGE OLD + NEW
    # =========================
    all_chunks = existing_chunks + new_chunks

    # =========================
    # SAVE CHUNKS
    # =========================
    os.makedirs("data/processed", exist_ok=True)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[INGEST] New chunks added: {len(new_chunks)}")
    print(f"[INGEST] Total chunks: {len(all_chunks)}")

    # =========================
    # UPDATE FAISS INDEX
    # =========================
    if new_chunks:
        print("[EMBED] Updating FAISS index...")
        update_faiss_index(new_chunks)
    else:
        print("[EMBED] No new chunks to embed")

    return len(new_chunks)
