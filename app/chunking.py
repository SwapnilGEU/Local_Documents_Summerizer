from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
from config import CHUNKS_PATH
import torch
import json

embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedding device:", embedding_device)

# The embedding model is needed either way (chunking OR loading a cached
# vectorstore downstream needs an embedding_function), so it's always loaded.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": embedding_device},
    encode_kwargs={"normalize_embeddings": True},
)

MIN_CHUNK_CHARS = 200
MAX_SEMANTIC_INPUT = 20000


def _load_cached_chunks():
    cached = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            cached.append(json.loads(line))
    return cached


def _build_chunks():
    # Only parse/ingest the source document when we actually need to
    # rebuild chunks - not on every import.
    from ingestion import sections

    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    merged_sections = []
    carry = ""

    for sec in sections:
        combined = (carry + "\n\n" + sec["text"]).strip() if carry else sec["text"]

        if len(combined) < MIN_CHUNK_CHARS:
            carry = combined
            continue

        merged_sections.append({**sec, "text": combined})
        carry = ""

    if carry:
        if merged_sections:
            merged_sections[-1]["text"] += "\n\n" + carry
        else:
            merged_sections.append({**sections[-1], "text": carry})

    print(
        f"Sections before merge: {len(sections)} -> "
        f"after merge: {len(merged_sections)}"
    )

    def chunk_section_text(section_text):
        pieces = (
            fallback_splitter.split_text(section_text)
            if len(section_text) > MAX_SEMANTIC_INPUT
            else [section_text]
        )

        chunks_out = []

        for piece in pieces:
            chunks_out.extend(semantic_splitter.split_text(piece))

        return chunks_out

    built = []

    for sec in merged_sections:
        for piece in chunk_section_text(sec["text"]):
            piece = piece.strip()

            if not piece:
                continue

            built.append({
                "chunk_id": len(built),
                "heading_path": sec["heading_path"],
                "page": sec["page"],
                "text": piece,
                "n_chars": len(piece),
            })

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in built:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(built):,} chunks to {CHUNKS_PATH}")

    return built


if CHUNKS_PATH.exists():
    print(f"Already chunked: {CHUNKS_PATH}")
    chunks = _load_cached_chunks()
    print(f"Loaded {len(chunks):,} cached chunks")
else:
    chunks = _build_chunks()