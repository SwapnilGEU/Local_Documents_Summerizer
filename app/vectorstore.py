from pathlib import Path
from chunking import chunks, embeddings
from config import CHROMA_PATH
from langchain_core.documents import Document
from langchain_chroma import Chroma

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
    print(f"Loaded existing ChromaDB from {CHROMA_PATH}")
else:
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="ml_book",
        persist_directory=CHROMA_PATH,
    )
    print("ChromaDB created successfully.")