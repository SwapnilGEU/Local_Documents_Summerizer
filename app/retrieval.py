from config import TOP_K_HYBRID
from vectorstore import vectorstore, documents

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_HYBRID},
)

from rank_bm25 import BM25Okapi

tokenized_docs = [
    doc.page_content.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(tokenized_docs)

print("BM25 index created.")

def hybrid_search(query, k=TOP_K_HYBRID, fetch_k=20):
    dense_docs = retriever.invoke(query)[:fetch_k]

    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = bm25_scores.argsort()[-fetch_k:][::-1]

    rrf_scores = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.metadata["chunk_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank + 1)

    for rank, idx in enumerate(bm25_indices):
        doc_id = documents[idx].metadata["chunk_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank + 1)

    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:k]

    doc_lookup = {
        doc.metadata["chunk_id"]: doc
        for doc in documents
    }

    return [doc_lookup[doc_id] for doc_id in ranked_ids]


if __name__ == "__main__":
    results = hybrid_search("What is machine learning?", k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Hybrid result {i} ---")
        print(doc.page_content[:350])