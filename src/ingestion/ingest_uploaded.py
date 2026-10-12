import os
import json

from src.ingestion.chunker import chunk_documents
from src.embeddings.update_index import update_faiss_index

DATA_PATH = "data/uploaded"
CHUNKS_PATH = "data/processed/chunks.json"


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path):
    import pymupdf

    with pymupdf.open(file_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def ingest_uploaded():
    print("[INGEST] Starting ingestion...")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

    files = sorted(os.listdir(DATA_PATH))
    print(f"[INGEST] Files found: {files}")

    # -------------------------
    # LOAD EXISTING CHUNKS
    # -------------------------
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    existing_by_doc = {}
    for chunk in existing_chunks:
        chunk_metadata = chunk.get("metadata", {})
        doc_id = chunk_metadata.get("doc_id") or chunk_metadata.get("source")
        if doc_id:
            existing_by_doc.setdefault(doc_id, []).append(chunk)

    # -------------------------
    # PROCESS FILES
    # -------------------------
    current_chunks_by_doc = {}
    for file in files:
        file_path = os.path.join(DATA_PATH, file)

        # skip empty files
        if os.path.getsize(file_path) == 0:
            print(f"[SKIP] Empty file: {file}")
            continue

        try:
            if file.lower().endswith(".pdf"):
                text = load_pdf(file_path)
            else:
                text = load_text(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load document {file}: {e}") from e

        if not text.strip():
            raise ValueError(f"No text could be extracted from document: {file}")

        docs = [{
            "text": text,
            "metadata": {
                "doc_id": file,
                "file_type": file.split(".")[-1]
            }
        }]

        chunks = chunk_documents(docs)

        print(f"[CHUNKER] {file} -> {len(chunks)} chunks")
        current_chunks_by_doc[file] = chunks

    def signature(chunks):
        return sorted(
            (
                (chunk.get("metadata", {}).get("chunk_id"), chunk.get("text", ""))
                for chunk in chunks
            ),
            key=lambda item: str(item[0]),
        )

    changed_docs = {
        doc_id
        for doc_id, chunks in current_chunks_by_doc.items()
        if signature(existing_by_doc.get(doc_id, [])) != signature(chunks)
    }
    replaced_docs = {
        doc_id
        for doc_id in changed_docs
        if existing_by_doc.get(doc_id)
    }
    new_chunks = [
        chunk
        for doc_id in sorted(changed_docs)
        for chunk in current_chunks_by_doc[doc_id]
    ]

    # -------------------------
    # MERGE + SAVE
    # -------------------------
    processed_docs = set(current_chunks_by_doc)
    all_chunks = [
        chunk
        for chunk in existing_chunks
        if (
            chunk.get("metadata", {}).get("doc_id")
            or chunk.get("metadata", {}).get("source")
        ) not in processed_docs
    ]
    all_chunks.extend(
        chunk
        for chunks in current_chunks_by_doc.values()
        for chunk in chunks
    )

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[INGEST] New chunks added: {len(new_chunks)}")
    print(f"[INGEST] Total chunks: {len(all_chunks)}")

    # -------------------------
    # UPDATE FAISS
    # -------------------------
    # Reconcile the index on every ingestion run. This also retries chunks
    # that were saved successfully when an earlier indexing attempt failed.
    update_faiss_index(all_chunks, rebuild_docs=replaced_docs)

    return len(new_chunks)