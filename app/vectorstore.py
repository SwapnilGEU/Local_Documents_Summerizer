from pathlib import Path

from chunking import chunks, embeddings
from config import CHROMA_PATH
from langchain_chroma import Chroma
from langchain_core.documents import Document

documents = [
    Document(
        page_content=chunk["text"],
        metadata={
            "chunk_id": chunk["chunk_id"],
            "heading_path": chunk["heading_path"],
            "page": chunk["page"],
        },
    )
    for chunk in chunks
]

print(f"Documents ready: {len(documents):,}")

_chroma_dir = Path(CHROMA_PATH)
_store_exists = _chroma_dir.exists() and any(_chroma_dir.iterdir())

if _store_exists:
    # Load the persisted collection instead of re-embedding everything.
    vectorstore = Chroma(
        collection_name="ml_book",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    _n_vectors = vectorstore._collection.count()
    if _n_vectors == len(documents):
        print(f"Loaded existing ChromaDB from {CHROMA_PATH} ({_n_vectors:,} vectors)")
    else:
        # e.g. 6,170 vectors for 1,234 chunks = every chunk stored 5 times,
        # which fills the top-k with copies of the same chunk.
        print(
            f"ChromaDB has {_n_vectors:,} vectors but there are {len(documents):,} chunks "
            "(duplicates or stale data) -> rebuilding collection"
        )
        vectorstore.delete_collection()
        _store_exists = False

if not _store_exists:
    # ids = chunk_id makes this idempotent: adding the same chunk again
    # overwrites it instead of creating a duplicate.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=[str(doc.metadata["chunk_id"]) for doc in documents],
        collection_name="ml_book",
        persist_directory=CHROMA_PATH,
    )
    print(f"ChromaDB created successfully ({vectorstore._collection.count():,} vectors).")
