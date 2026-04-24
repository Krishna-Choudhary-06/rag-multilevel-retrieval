from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import os
import shutil

from src.pipeline.rag_pipeline import RAGPipeline
from src.ingestion.ingest_uploaded import ingest_uploaded

app = FastAPI()

@app.on_event("startup")
def startup_event():
    try:
        print("[STARTUP] Running initial ingestion...")
        num_chunks = ingest_uploaded()
        print(f"[STARTUP] Ingestion complete. Chunks: {num_chunks}")
    except Exception as e:
        print(f"[STARTUP ERROR] Ingestion failed: {e}")

# 🔥 FORCE RELOAD PIPELINE
global rag
from src.memory.session_manager import SessionManager

session_manager = SessionManager()

UPLOAD_DIR = "data/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================
# REQUEST MODEL
# =========================
from typing import Optional, Dict


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"
    filters: Optional[Dict] = None
    mode: str = "Auto"


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"message": "RAG API running"}


# =========================
# QUERY ENDPOINT
# =========================
@app.post("/query")
def query_rag(req: QueryRequest):

    user_id = req.user_id if hasattr(req, "user_id") else "default"

    rag = session_manager.get_pipeline(user_id)

    response = rag.run(
    req.query,
    user_id=req.user_id,
    filters=req.filters,
    mode=req.mode
    )
    return {
        "query": req.query,
        "mode": req.mode,
        "answer": response.get("answer", ""),
        "sources": response.get("sources", []),
        "scores": response.get("scores", []),
        "reranked": response.get("reranked", []),
        "retrieved_chunks": response.get("retrieved_chunks", []),
        "latency": response.get("latency", 0),
    }


# =========================
# SINGLE FILE UPLOAD
# =========================
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):

    saved_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        saved_files.append(file_path)

    # ingest
    num_chunks = ingest_uploaded()

    # 🔥 CRITICAL FIX: Clear sessions to force reload
    session_manager.clear_sessions()

    return {
        "message": "Files uploaded and indexed successfully",
        "files": saved_files,
        "chunks_created": num_chunks,
    }


# =========================
# MULTIPLE FILE UPLOAD
# =========================
@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    saved = []

    for file in files:
        path = os.path.join(UPLOAD_DIR, file.filename)

        with open(path, "wb") as f:
            content = await file.read()
            f.write(content)

        saved.append(file.filename)

    num_chunks = ingest_uploaded()
    session_manager.clear_sessions()

    return {
        "message": "Files uploaded & indexed",
        "files": saved,
        "chunks_created": num_chunks,
    }


# =========================
# RESET INDEX
# =========================
@app.post("/reset")
def reset_index():
    shutil.rmtree("src/embeddings/faiss_index", ignore_errors=True)

    return {"message": "Index cleared"}


import faiss
import os
import json


@app.get("/status")
def system_status():
    index_path = "src/embeddings/faiss_index/index.faiss"
    meta_path = "src/embeddings/faiss_index/metadata.json"

    # FAISS info
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        total_vectors = index.ntotal
    else:
        total_vectors = 0

    # Metadata info
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        total_chunks = len(metadata)
    else:
        total_chunks = 0

    return {
        "total_vectors": total_vectors,
        "total_chunks": total_chunks,
        "index_exists": os.path.exists(index_path),
        "metadata_exists": os.path.exists(meta_path),
    }


@app.post("/ingest-folder")
def ingest_folder_api(path: str):
    from src.ingestion.ingest_folder import ingest_folder

    chunks = ingest_folder(path)

    return {"message": "Folder ingested", "chunks_created": chunks}
