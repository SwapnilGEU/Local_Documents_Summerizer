import sys
import time
from pathlib import Path
from uuid import uuid4

# Make app/'s own folder importable directly, matching how app/*.py
# imports itself internally (bare `from config import ...` etc.)
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.rag import rag
from app.logging_utils import log_event
from app.metrics import metrics


app = FastAPI(title="Advanced RAG API")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Advanced RAG API is running"}


@app.post("/query")
def query_rag(request: QueryRequest, http_request: Request):
    request_id = http_request.headers.get("X-Request-ID") or str(uuid4())
    request_start = time.perf_counter()

    log_event(
        "request_started",
        request_id=request_id,
        query=request.question,
        endpoint="/query",
        method="POST",
        query_length=len(request.question),
    )

    try:
        answer, sources, run_metrics = rag(request.question, request_id=request_id)

        latency_ms = (time.perf_counter() - request_start) * 1000
        metrics.record_request(latency_ms, success=True)
        metrics.save_snapshot()

        log_event(
            "request_completed",
            request_id=request_id,
            status_code=200,
        )

        # Structured per-request JSON: data_logs/<date>/metrics/<id>.json
        metrics.save_request_snapshot(
            request_id,
            request_section={
                "endpoint": "/query",
                "question": request.question,
                "status": "success",
                "total_latency_ms": round(latency_ms, 2),
            },
            rag_section=run_metrics["rag_section"],
            llm_section=run_metrics["llm_section"],
        )

        return {
            "request_id": request_id,
            "question": request.question,
            "answer": answer,
            "sources": [doc.metadata for doc in sources],
        }
    except Exception:
        latency_ms = (time.perf_counter() - request_start) * 1000
        metrics.record_request(latency_ms, success=False)
        metrics.save_snapshot()

        log_event(
            "request_failed",
            request_id=request_id,
            endpoint="/query",
        )

        metrics.save_request_snapshot(
            request_id,
            request_section={
                "endpoint": "/query",
                "question": request.question,
                "status": "failed",
                "total_latency_ms": round(latency_ms, 2),
            },
            rag_section={},
            llm_section={},
        )
        raise


@app.get("/metrics")
def get_metrics():
    return metrics.summary()
