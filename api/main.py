import sys
from pathlib import Path

# Make app/'s own folder importable directly, matching how app/*.py
# imports itself internally (bare `from config import ...` etc.)
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import rag  # bare import, not app.rag


app = FastAPI(title="Advanced RAG API")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Advanced RAG API is running"}


@app.post("/query")
def query_rag(request: QueryRequest):
    answer, sources = rag(request.question)

    return {
        "question": request.question,
        "answer": answer,
        "sources": [doc.metadata for doc in sources],
    }