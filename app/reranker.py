from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL,TOP_K_FINAL,TOP_K_HYBRID
from retrieval import hybrid_search


reranker = CrossEncoder(RERANKER_MODEL)

def rerank(query, docs, top_k=TOP_K_FINAL):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_k]]


if __name__ == "__main__":
    query = "What is machine learning?"

    hybrid_docs = hybrid_search(query, k=TOP_K_HYBRID)
    final_docs = rerank(query, hybrid_docs, top_k=TOP_K_FINAL)

    '''for i, doc in enumerate(final_docs, 1):
        print(f"\n--- Reranked result {i} ---")
        print(doc.page_content[:350])'''