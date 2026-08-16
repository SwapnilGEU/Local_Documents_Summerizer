from deepeval.models import OllamaModel
import pandas as pd
from deepeval.test_case import LLMTestCase
from eval_set import eval_rows
from app.config import LOCAL_MODEL,OLLAMA_BASE_URL

evaluator_model = OllamaModel(
    model=LOCAL_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

print("DeepEval evaluator:", LOCAL_MODEL)
print("Base URL:", OLLAMA_BASE_URL)

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

faithfulness_metric = FaithfulnessMetric(
    model=evaluator_model,
    threshold=0.5,
    include_reason=True,
)

answer_relevancy_metric = AnswerRelevancyMetric(
    model=evaluator_model,
    threshold=0.5,
    include_reason=True,
)

contextual_precision_metric = ContextualPrecisionMetric(
    model=evaluator_model,
    threshold=0.5,
    include_reason=True,
)

contextual_recall_metric = ContextualRecallMetric(
    model=evaluator_model,
    threshold=0.5,
    include_reason=True,
)

metrics = {
    "faithfulness": faithfulness_metric,
    "answer_relevancy": answer_relevancy_metric,
    "contextual_precision": contextual_precision_metric,
    "contextual_recall": contextual_recall_metric,
}

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
        "answer": row["response"],  # <-- added
    }

    for name, metric in metrics.items():
        metric.measure(test_case)
        row_result[name] = float(metric.score)

    results.append(row_result)

results_df = pd.DataFrame(results)
results_df

score_columns = [
    "faithfulness",
    "answer_relevancy",
    "contextual_precision",
    "contextual_recall",
]

results_df[score_columns].mean().sort_values(ascending=False)
