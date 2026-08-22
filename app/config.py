from pathlib import Path
import json
import re
import numpy as np
import os
RAW_PDF = Path("data/raw/MachineLearningTomMitchell.pdf")
MD_PATH = Path("data/processed/ml_book.md")
CLEAN_PATH = Path("data/processed/clean_ml_book.md")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CHROMA_PATH = "data/chroma"
MAX_VALIDATION_RETRIES = 2

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Ollama model used for both generation and DeepEval judging.
LOCAL_MODEL = "qwen3:4b-instruct"
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

TOP_K_HYBRID = 8
TOP_K_FINAL = 4

MD_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Configuration loaded.")
print("LLM:", LOCAL_MODEL)
