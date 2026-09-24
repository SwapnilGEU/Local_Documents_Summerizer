"""Central settings for all benchmark scripts. Edit this file, not the scripts."""

import os
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
APP_DIR = REPO_ROOT / "app"
RESULTS_DIR = BENCH_DIR / "results"
GOLDEN_PATH = BENCH_DIR / "golden_set.jsonl"
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"

# ---------------- Retrieval benchmark ----------------
# Order = the order you plan to build them. Only the ones you list in
# --methods (or ENABLED_METHODS) are loaded, so BM25 works on its own.
ENABLED_METHODS = ["bm25", "chroma", "hybrid", "rerank"]  # optional extra: "chroma_mmr"
K_VALUES = [1, 3, 5, 8]  # metrics are reported at each k
FETCH_K = 20  # how many candidates each retriever returns
RERANK_CANDIDATES = 20  # how many hybrid results the cross-encoder re-scores

# ---------------- LLM speed benchmark ----------------
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ~3-4B models from different companies, all fit in 6 GB VRAM at Q4.
# `think` is only sent for models that support the flag.
MODELS = [
    {"name": "qwen3:4b-instruct", "think": None},  # Alibaba (your baseline)
    {"name": "gemma3:4b", "think": None},  # Google
    {"name": "llama3.2:3b", "think": None},  # Meta
    {"name": "phi4-mini:3.8b", "think": None},  # Microsoft
    {"name": "granite4:micro", "think": None},  # IBM
]

GEN_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 4096,  # keep identical across models or the comparison is unfair
    "num_predict": 256,
}
N_PROMPTS = 10  # how many golden questions to use as RAG prompts
REPEATS = 3  # runs per prompt; we report the median
CONTEXT_CHUNKS = 4  # chunks stuffed into each prompt (matches TOP_K_FINAL)
