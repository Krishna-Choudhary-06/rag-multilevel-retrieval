import os
import json
import faiss
import numpy as np
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
        nlist = 100

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
def update_faiss_index(use_ivf=False):
    chunks = load_chunks()

    if not chunks:
        print("[EMBED] No chunks found")
        return

    index, metadata = load_index_and_meta()

    # =========================
    # REMOVE DUPLICATES
    # =========================
    existing_texts = set([m["text"] for m in metadata])

    new_chunks = [c for c in chunks if c["text"] not in existing_texts]

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

    # =========================
    # UPDATE METADATA
    # =========================
    metadata.extend(new_chunks)

    # =========================
    # SAVE INDEX + META
    # =========================
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[EMBED] Index updated successfully")
