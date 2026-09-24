import time

from config import MAX_VALIDATION_RETRIES, TOP_K_FINAL, TOP_K_HYBRID
from llm import local_llm
from logging_utils import log_event
from metrics import metrics
from pydantic import ValidationError
from reranker import rerank
from retrieval import hybrid_search
from schemas import RAGAnswer


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


def _call_llm_once(prompt_text):
    """Single raw LLM call. Returns (raw_text, response_obj, latency_ms,
    prompt_tokens, completion_tokens, tokens_per_second)."""

    llm_start = time.perf_counter()
    response = local_llm.invoke(prompt_text)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    raw_text = extract_response_text(response)

    meta = response.response_metadata or {}
    prompt_tokens = meta.get("prompt_eval_count", 0)
    completion_tokens = meta.get("eval_count", 0)
    eval_duration_s = meta.get("eval_duration", 0) / 1e9
    tokens_per_second = (
        completion_tokens / eval_duration_s if eval_duration_s > 0 else 0.0
    )

    return (
        raw_text,
        response,
        llm_ms,
        prompt_tokens,
        completion_tokens,
        tokens_per_second,
    )


def generate_validated_answer(prompt_text, request_id=None):
    """Calls the LLM, validates the answer against RAGAnswer, and retries
    on validation failure up to MAX_VALIDATION_RETRIES times. Falls back
    to a graceful failure message if every attempt fails validation.

    Returns:
        answer (str), last_response (obj), total_llm_ms (float),
        attempts_used (int), llm_totals (dict of token/latency stats)
    """
    original_prompt = prompt_text
    total_llm_ms = 0.0
    attempt = 0
    last_response = None

    llm_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens_per_second_last": 0.0,
    }

    while True:
        raw_text, response, llm_ms, prompt_tokens, completion_tokens, tps = (
            _call_llm_once(prompt_text)
        )
        last_response = response
        total_llm_ms += llm_ms

        metrics.record_llm(llm_ms)
        metrics.record_tokens(prompt_tokens, completion_tokens, tps)

        llm_totals["prompt_tokens"] += prompt_tokens
        llm_totals["completion_tokens"] += completion_tokens
        llm_totals["tokens_per_second_last"] = tps

        log_event(
            "llm_completed",
            request_id=request_id,
            attempt=attempt + 1,
            latency_ms=round(llm_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=round(tps, 2),
        )

        try:
            validated = RAGAnswer(answer=raw_text)
            return validated.answer, last_response, total_llm_ms, attempt, llm_totals

        except ValidationError as exc:
            metrics.record_validation_failure()

            log_event(
                "rag_validation_error",
                request_id=request_id,
                attempt=attempt + 1,
                error=str(exc),
            )

            attempt += 1

            if attempt > MAX_VALIDATION_RETRIES:
                metrics.record_graceful_failure()

                log_event(
                    "rag_graceful_failure",
                    request_id=request_id,
                    attempts=attempt,
                    reason="LLM response failed Pydantic validation",
                )

                fallback = (
                    "I'm sorry, but I was unable to generate a valid "
                    "answer from the provided documents."
                )
                return fallback, last_response, total_llm_ms, attempt, llm_totals

            metrics.record_retry()

            prompt_text = f"""
{original_prompt}

IMPORTANT:
Your previous response failed validation.

Validation error:
{exc}

Generate the answer again.
Return a non-empty answer.
Follow all the original instructions.
Do not mention this validation failure.
"""


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

    answer, _response, total_llm_ms, validation_attempts, llm_totals = (
        generate_validated_answer(prompt_text, request_id=request_id)
    )

    total_ms = (time.perf_counter() - total_start) * 1000

    log_event(
        "rag_completed",
        request_id=request_id,
        total_latency_ms=round(total_ms, 2),
    )

    # Structured per-request metrics record (REQUEST / RAG / LLM sections).
    # api/main.py fills in the "request" section (status, endpoint) and
    # persists this via metrics.save_request_snapshot(...).
    run_metrics = {
        "rag_section": {
            "total_latency_ms": round(total_ms, 2),
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "documents_retrieved": len(hybrid_docs),
            "reranking_latency_ms": round(rerank_ms, 2),
            "documents_selected": len(final_docs),
        },
        "llm_section": {
            "total_llm_latency_ms": round(total_llm_ms, 2),
            "validation_attempts": validation_attempts,
            "graceful_failure": validation_attempts > MAX_VALIDATION_RETRIES,
            "prompt_tokens": llm_totals["prompt_tokens"],
            "completion_tokens": llm_totals["completion_tokens"],
            "tokens_per_second": round(llm_totals["tokens_per_second_last"], 2),
        },
    }

    return answer, final_docs, run_metrics


if __name__ == "__main__":
    question = "What is machine learning?"
    answer, sources, run_metrics = rag(question)

    print("QUESTION:")
    print(question)
    print("\nANSWER:")
    print(answer)
    print("\nSOURCES:")
    for i, doc in enumerate(sources, 1):
        print(f"{i}. {doc.metadata}")
