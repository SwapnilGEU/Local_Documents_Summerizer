import sys
from pathlib import Path

# Make `app/` importable both as a package (evaluation/*) and internally
# (app/*.py use bare imports like `from config import ...`).
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

EVAL_SET = [
    {
        "question": "What is machine learning?",
        "reference": (
            "Machine learning is the study of computer programs that "
            "improve their performance on a task through experience."
        ),
    },
    {
        "question": "What is supervised learning?",
        "reference": (
            "Supervised learning is learning a function that maps inputs "
            "to outputs from a set of labeled training examples."
        ),
    },
    {
        "question": "What is unsupervised learning?",
        "reference": (
            "Unsupervised learning is learning patterns or structure "
            "from data that has no labeled outputs."
        ),
    },
    {
        "question": "What is reinforcement learning?",
        "reference": (
            "Reinforcement learning is learning what actions to take, "
            "given a state, in order to maximize a numerical reward signal over time."
        ),
    },
    {
        "question": "What are the main applications of machine learning?",
        "reference": (
            "Machine learning is applied in areas such as data mining, "
            "speech and image recognition, fraud detection, autonomous "
            "vehicles, and information-filtering systems."
        ),
    },
]


def build_eval_rows(eval_set=EVAL_SET, verbose=True):
    """
    Runs the RAG pipeline over each eval question and builds eval rows.
    This is the expensive part (one LLM call per question) - it's a
    function now, not import-time code, so importing this module is free.
    """
    from app.rag import rag  # imported lazily so a plain `import eval_set` is cheap

    rows = []

    for item in eval_set:
        if verbose:
            print(f"Running RAG for: {item['question']}")

        try:
            answer, sources = rag(item["question"])
        except Exception as e:
            print(f"  Skipped (RAG call failed: {e})")
            continue

        rows.append({
            "user_input": item["question"],
            "response": answer,
            "retrieved_contexts": [doc.page_content for doc in sources],
            "reference": item["reference"],
        })

    if verbose:
        print(f"Evaluation samples: {len(rows)}")

    return rows


if __name__ == "__main__":
    eval_rows = build_eval_rows()
    for row in eval_rows:
        print(f"\nQ: {row['user_input']}\nA: {row['response'][:200]}")