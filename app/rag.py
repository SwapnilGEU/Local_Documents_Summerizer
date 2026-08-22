import time

from config import TOP_K_FINAL, TOP_K_HYBRID
from retrieval import hybrid_search
from llm import local_llm
from reranker import rerank
from logging_utils import log_event
from metrics import metrics


def build_context(docs):
    parts = []

    for doc in docs:
        meta = doc.metadata
        source = (
            f"[{meta.get('heading_path', 'Unknown section')}, "
            f"p.{meta.get('page', '?')}]"
        )
        parts.append(f"{source}\n{doc.page_content}")

    return "\n\n".join(parts)


def build_prompt(context, question):
    return f"""<|system|>
You are a helpful question-answering assistant for machine learning.

Answer the question using ONLY the supplied context.

Give a clear and sufficiently detailed answer. Use multiple sentences
when the context provides useful supporting information.

Do not add information that is not supported by the context.

Cite the relevant source tag at the end of the answer when appropriate.

If the answer is not present in the context, say:
"I don't know based on the provided context."

<|user|>
Context:
{context}

Question:
{question}

<|assistant|>
"""


def extract_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []

        for block in content:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])

        return "".join(texts).strip()

    return str(content).strip()


def rag(query, request_id=None):
    request_id = request_id or "local"
    total_start = time.perf_counter()

    log_event(
        "rag_started",
        request_id=request_id,
        query_length=len(query),
    )

    start = time.perf_counter()
    hybrid_docs = hybrid_search(
        query,
        k=TOP_K_HYBRID,
        fetch_k=20,
    )
    retrieval_ms = (time.perf_counter() - start) * 1000
    metrics.record_retrieval(retrieval_ms)

    log_event(
        "retrieval_completed",
        request_id=request_id,
        latency_ms=round(retrieval_ms, 2),
        documents_retrieved=len(hybrid_docs),
    )

    start = time.perf_counter()
    final_docs = rerank(
        query,
        hybrid_docs,
        top_k=TOP_K_FINAL,
    )
    rerank_ms = (time.perf_counter() - start) * 1000
    metrics.record_reranking(rerank_ms)

    log_event(
        "reranking_completed",
        request_id=request_id,
        latency_ms=round(rerank_ms, 2),
        documents_selected=len(final_docs),
    )

    context = build_context(final_docs)
    prompt_text = build_prompt(context, query)

    start = time.perf_counter()
    response = local_llm.invoke(prompt_text)
    llm_ms = (time.perf_counter() - start) * 1000
    answer = extract_response_text(response)

    metrics.record_llm(llm_ms)

    meta = response.response_metadata
    prompt_tokens = meta.get("prompt_eval_count", 0)
    completion_tokens = meta.get("eval_count", 0)
    eval_duration_s = meta.get("eval_duration", 0) / 1e9
    tokens_per_second = (
        completion_tokens / eval_duration_s if eval_duration_s > 0 else 0.0
    )
    metrics.record_tokens(prompt_tokens, completion_tokens, tokens_per_second)

    log_event(
        "llm_completed",
        request_id=request_id,
        latency_ms=round(llm_ms, 2),
        context_chars=len(context),
        prompt_chars=len(prompt_text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=round(tokens_per_second, 2),
    )

    total_ms = (time.perf_counter() - total_start) * 1000
    log_event(
        "rag_completed",
        request_id=request_id,
        total_latency_ms=round(total_ms, 2),
    )

    return answer, final_docs


if __name__ == "__main__":
    question = "What is machine learning?"
    answer, sources = rag(question)

    print("QUESTION:")
    print(question)
    print("\nANSWER:")
    print(answer)
    print("\nSOURCES:")
    for i, doc in enumerate(sources, 1):
        print(f"{i}. {doc.metadata}")
