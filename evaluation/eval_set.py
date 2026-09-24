"""
Small end-to-end eval set: 5 questions whose reference answers come ONLY
from data/processed/chunks.jsonl (Mitchell, *Machine Learning*).

Why this matters: DeepEval's contextual recall and faithfulness compare the
answer and the retrieved chunks against the reference. If a reference says
something the book never says, the pipeline gets punished for being
correctly grounded. The old set had exactly that problem: the book never
uses the term "supervised learning" and only mentions unsupervised
clustering in passing, so Q2/Q3 are reworded to what the book actually covers.

Each row stores:
    relevant_chunk_ids  chunks that contain the answer (checked against retrieval)
    evidence            verbatim quote from the first relevant chunk
                        (validate_eval_set() fails if re-chunking moved it)
    reference           the expected answer, paraphrased from those chunks
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"

EVAL_SET = [
    {
        "qid": "e1",
        "question": "What is machine learning?",
        "relevant_chunk_ids": [9, 0],
        "evidence": "is said to learn from experience E with respect to some class of tasks",
        "reference": (
            "Machine learning is the study of computer programs that automatically "
            "improve with experience. Formally, a program learns from experience E "
            "with respect to some class of tasks T and performance measure P if its "
            "performance at tasks in T, as measured by P, improves with experience E."
        ),
    },
    {
        "qid": "e2",
        "question": "What is concept learning from labeled training examples?",
        "relevant_chunk_ids": [65, 64],
        "evidence": "Inferring **a** boolean-valued function from training examples of its input and output",
        "reference": (
            "Concept learning is automatically inferring the general definition of a "
            "concept from examples labeled as members or nonmembers of that concept, "
            "i.e. approximating a boolean-valued function from training examples of "
            "its input and output."
        ),
    },
    {
        "qid": "e3",
        "question": "How can unsupervised clustering be used to place the kernel centers of a radial basis function network?",
        "relevant_chunk_ids": [696],
        "evidence": "unsupervised clustering algorithms that fit the training instances",
        "reference": (
            "Prototypical clusters of the training instances are found with unsupervised "
            "clustering, fitting the instances (but not their target values) to a "
            "mixture of Gaussians, for example with the EM algorithm, and a kernel "
            "function is centered at each cluster. The target values are only used "
            "afterwards to set the output layer weights."
        ),
    },
    {
        "qid": "e4",
        "question": "What is reinforcement learning?",
        "relevant_chunk_ids": [1093, 1094, 1143],
        "evidence": "Reinforcement learning addresses the question of how an autonomous agent",
        "reference": (
            "Reinforcement learning addresses how an autonomous agent that senses and "
            "acts in its environment can learn to choose optimal actions to achieve its "
            "goals. A trainer provides rewards or penalties for the resulting states, and "
            "the agent learns from this indirect, delayed reward to choose sequences of "
            "actions that produce the greatest cumulative reward."
        ),
    },
    {
        "qid": "e5",
        "question": "What are the main applications of machine learning?",
        "relevant_chunk_ids": [0, 9, 10, 11],
        "evidence": "learn to detect fraudulent credit card transactions",
        "reference": (
            "Examples include data-mining programs that detect fraudulent credit card "
            "transactions, information-filtering systems that learn users' reading "
            "preferences, autonomous vehicles that learn to drive on public highways "
            "(ALVINN), speech recognition (SPHINX), classifying celestial objects in "
            "large sky-survey databases (NASA), and world-class backgammon playing "
            "(TD-GAMMON)."
        ),
    },
]


def validate_eval_set(eval_set=EVAL_SET):
    """Returns a list of problems (empty = OK). Run after any re-chunking."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = {c["chunk_id"]: c for c in map(json.loads, f)}
    problems = []
    for item in eval_set:
        missing = [cid for cid in item["relevant_chunk_ids"] if cid not in chunks]
        if missing:
            problems.append(f"{item['qid']}: chunks {missing} do not exist")
            continue
        primary = chunks[item["relevant_chunk_ids"][0]]["text"].lower()
        if item["evidence"].lower() not in primary:
            problems.append(f"{item['qid']}: evidence not found in chunk {item['relevant_chunk_ids'][0]}")
    return problems


def build_eval_rows(eval_set=EVAL_SET, verbose=True):
    """
    Runs the full RAG pipeline (hybrid search -> rerank -> LLM) on each
    question. Uses whatever model app/config.py resolved LOCAL_MODEL to.
    """
    import time

    from rag import rag  # bare import: same module copy the app uses

    rows = []
    for item in eval_set:
        if verbose:
            print(f"Running RAG for {item['qid']}: {item['question']}")

        start = time.perf_counter()
        try:
            answer, sources, run_metrics = rag(item["question"], request_id=f"eval-{item['qid']}")
        except Exception as e:  # noqa: BLE001 - one failed question must not stop the eval run
            print(f"  Skipped (RAG call failed: {e})")
            continue
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [doc.metadata["chunk_id"] for doc in sources]
        relevant = set(item["relevant_chunk_ids"])
        rows.append(
            {
                "qid": item["qid"],
                "user_input": item["question"],
                "response": answer,
                "reference": item["reference"],
                "retrieved_contexts": [doc.page_content for doc in sources],
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                # did the final (reranked) context contain at least one gold chunk?
                "retrieval_hit": bool(relevant & set(retrieved_ids)),
                "latency_ms": round(latency_ms, 1),
                "llm": run_metrics.get("llm_section", {}),
            }
        )

    if verbose:
        print(f"Evaluation samples: {len(rows)}")
    return rows


if __name__ == "__main__":
    problems = validate_eval_set()
    print("\n".join(problems) if problems else f"OK - {len(EVAL_SET)} eval questions match chunks.jsonl")
