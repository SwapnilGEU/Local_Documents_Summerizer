import re

from config import TOP_K_HYBRID
from rank_bm25 import BM25Okapi
from vectorstore import documents, vectorstore

# Kept for anything that still imports it; hybrid_search asks the vector store
# directly so it can fetch fetch_k results instead of a fixed TOP_K_HYBRID.
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_HYBRID},
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase and keep only letters/digits.

    The chunks come from PDF -> markdown, so they contain things like
    '**_entropy,_**'. A plain .split() keeps that as one token that never
    matches the query word 'entropy'. Use this SAME function for chunks and
    queries, otherwise BM25 compares apples to oranges.
    """
    return _TOKEN_RE.findall(text.lower())


tokenized_docs = [tokenize(doc.page_content) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
doc_lookup = {doc.metadata["chunk_id"]: doc for doc in documents}

print("BM25 index created.")


def hybrid_search(query, k=TOP_K_HYBRID, fetch_k=20):
    # Dense: ask for fetch_k results (previously capped at TOP_K_HYBRID=8,
    # so the [:fetch_k] slice never actually got 20 candidates).
    dense_docs = vectorstore.similarity_search(query, k=fetch_k)

    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_indices = bm25_scores.argsort()[-fetch_k:][::-1]

    # Reciprocal Rank Fusion: each list votes 1/(60 + rank).
    rrf_scores = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.metadata["chunk_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank + 1)

    for rank, idx in enumerate(bm25_indices):
        doc_id = documents[idx].metadata["chunk_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank + 1)

    ranked_ids = sorted(rrf_scores, key=lambda doc_id: rrf_scores[doc_id], reverse=True)[:k]

    return [doc_lookup[doc_id] for doc_id in ranked_ids]


if __name__ == "__main__":
    results = hybrid_search("What is machine learning?", k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Hybrid result {i} ---")
        print(doc.page_content[:350])
