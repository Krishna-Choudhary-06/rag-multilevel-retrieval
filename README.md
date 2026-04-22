# Multi-Level RAG System (Advanced Retrieval Pipeline)

A production-style Retrieval-Augmented Generation (RAG) system designed for efficient, scalable, and high-accuracy document retrieval and response generation.

---

## 🚀 Features

- Multi-stage retrieval pipeline
- Metadata filtering (Level 1)
- Vector search using FAISS (Level 2)
- Hybrid search (BM25 + embeddings)
- Cross-encoder reranking (Level 3)
- Query rewriting & expansion
- Local LLM integration (Ollama)
- FastAPI backend
- Streamlit UI
- Query caching (performance optimization)
- Evaluation pipeline

---

## 🧠 Architecture

User Query  
→ Query Rewriting  
→ Hybrid Search (BM25 + Vector)  
→ Merge Results  
→ Reranking  
→ Context Selection  
→ LLM Generation  
→ Final Answer  

---

## 🖥️ Demo

### Example Query:
"What is retrieval augmented generation?"

### Output:
- Accurate contextual answer
- Source documents returned

---

## ⚙️ Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt