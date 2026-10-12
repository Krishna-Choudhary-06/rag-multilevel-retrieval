import os
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer

# =========================
# PATHS
# =========================
CHUNKS_PATH = "data/processed/chunks.json"
INDEX_DIR = "src/embeddings/faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")

# Ensure directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# =========================
# DEVICE SETUP
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[EMBED] Using device: {device}")

# =========================
# LOAD MODEL
# =========================
model = SentenceTransformer("BAAI/bge-small-en", device=device)


# =========================
# LOAD CHUNKS
# =========================
def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        return []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# LOAD EXISTING INDEX + META
# =========================
def load_index_and_meta():
    if os.path.exists(INDEX_PATH):
        print("[FAISS] Loading existing index...")
        index = faiss.read_index(INDEX_PATH)
    else:
        print("[FAISS] No index found. Creating new...")
        index = None

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    return index, metadata


# =========================
# CREATE INDEX (AUTO TYPE)
# =========================
def create_index(embeddings, use_ivf=False):
    dim = embeddings.shape[1]

    if use_ivf:
        print("[FAISS] Creating IVF index...")
        nlist = min(100, len(embeddings))

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)

        index.train(embeddings)
    else:
        print("[FAISS] Creating Flat index...")
        index = faiss.IndexFlatL2(dim)

    return index


# =========================
# UPDATE INDEX
# =========================
def chunk_identity(chunk):
    metadata = chunk.get("metadata", {})
    doc_id = metadata.get("doc_id") or metadata.get("source")
    chunk_id = metadata.get("chunk_id")

    if chunk_id is None and doc_id and chunk.get("id"):
        legacy_id = str(chunk["id"])
        prefix = f"{doc_id}_"
        if legacy_id.startswith(prefix):
            suffix = legacy_id[len(prefix):]
            if suffix.isdigit():
                chunk_id = int(suffix)

    if doc_id and chunk_id is not None:
        return ("document_chunk", doc_id, chunk_id)
    if doc_id:
        return ("document_text", doc_id, chunk.get("text", ""))
    return ("text", chunk.get("text", ""))


def update_faiss_index(chunks=None, use_ivf=False, rebuild_docs=None):
    requested_chunks = load_chunks() if chunks is None else chunks
    if not requested_chunks:
        print("[EMBED] No chunks found")
        return

    index, metadata = load_index_and_meta()
    rebuild_docs = set(rebuild_docs or ())

    rebuild = bool(rebuild_docs) or index is None or index.ntotal != len(metadata)
    if rebuild:
        print("[FAISS] Rebuilding index to restore vector/metadata alignment...")
        preserved_metadata = [
            item
            for item in metadata
            if (item.get("metadata", {}).get("doc_id")
                or item.get("metadata", {}).get("source"))
            not in rebuild_docs
        ]
        chunks = requested_chunks + preserved_metadata
        metadata = []
        index = None
    else:
        chunks = requested_chunks

    existing_keys = {chunk_identity(item) for item in metadata}
    new_chunks = []
    for chunk in chunks:
        key = chunk_identity(chunk)
        if key not in existing_keys:
            new_chunks.append(chunk)
            existing_keys.add(key)

    if not new_chunks:
        print("[FAISS] No new chunks to add")
        return

    print(f"[EMBED] Encoding NEW chunks only: {len(new_chunks)}")

    texts = [c["text"] for c in new_chunks]

    # =========================
    # BATCH EMBEDDING (FAST)
    # =========================
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    # =========================
    # CREATE INDEX IF NEEDED
    # =========================
    if index is None:
        index = create_index(embeddings, use_ivf)

    # =========================
    # ADD VECTORS
    # =========================
    if isinstance(index, faiss.IndexIVF) and not index.is_trained:
        print("[FAISS] Training IVF index...")
        index.train(embeddings)

    index.add(embeddings)

    print(f"[FAISS] Added new vectors: {len(embeddings)}")

    metadata.extend(new_chunks)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[EMBED] Index updated successfully")
