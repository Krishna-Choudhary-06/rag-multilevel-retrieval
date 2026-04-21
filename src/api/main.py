from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.rag_pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI()

# Initialize RAG pipeline once (important for performance)
rag = RAGPipeline()


# Request schema
class QueryRequest(BaseModel):
    query: str


# Health check route
@app.get("/")
def home():
    return {"message": "RAG API is running"}


# Main RAG endpoint
@app.post("/query")
def query_rag(req: QueryRequest):
    response = rag.run(req.query)

    return {
        "query": req.query,
        "answer": response["answer"],
        "sources": response["sources"]
    }