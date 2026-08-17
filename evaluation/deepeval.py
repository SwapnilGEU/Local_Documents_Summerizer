import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from config import LOCAL_MODEL, OLLAMA_BASE_URL
from llm import check_ollama_connection
from eval_set import build_eval_rows

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.csv"


def build_metrics(evaluator_model):
    return {
        "faithfulness": FaithfulnessMetric(
            model=evaluator_model, threshold=0.5, include_reason=True,
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            model=evaluator_model, threshold=0.5, include_reason=True,
        ),
        "contextual_precision": ContextualPrecisionMetric(
            model=evaluator_model, threshold=0.5, include_reason=True,
        ),
        "contextual_recall": ContextualRecallMetric(
            model=evaluator_model, threshold=0.5, include_reason=True,
        ),
    }


def run_evaluation(eval_rows):
    evaluator_model = OllamaModel(
        model=LOCAL_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    print("DeepEval evaluator:", LOCAL_MODEL)
    print("Base URL:", OLLAMA_BASE_URL)

    metrics = build_metrics(evaluator_model)
    print("DeepEval metrics ready.")

    results = []

    for row in eval_rows:
        print(f"Evaluating: {row['user_input']}")

        test_case = LLMTestCase(
            input=row["user_input"],
            actual_output=row["response"],
            expected_output=row["reference"],
            retrieval_context=row["retrieved_contexts"],
        )

        row_result = {
            "question": row["user_input"],
            "answer": row["response"],
        }

        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                row_result[name] = float(metric.score)
            except Exception as e:
                print(f"  {name} failed: {e}")
                row_result[name] = None

        results.append(row_result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    if not check_ollama_connection():
        raise RuntimeError(
            f"Ollama server not reachable at {OLLAMA_BASE_URL}.\n"
            "Start it with `ollama serve`, or open the Ollama desktop app, "
            "then re-run this script."
        )

    eval_rows = build_eval_rows()

    if not eval_rows:
        raise RuntimeError("No eval rows were produced - check the RAG pipeline.")

    results_df = run_evaluation(eval_rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")

    score_columns = [
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
    ]
    print("\nMean scores:")
    print(results_df[score_columns].mean().sort_values(ascending=False))