# 🧪 evaluation/

End-to-end **answer quality** checks for the PDFinsight RAG pipeline.

The benchmarks in [`../benchmarks`](../benchmarks/README.md) answer *"did we retrieve the right chunk?"* and *"how fast is the model?"*.
This folder answers the question that matters most to a user: **"Is the final answer correct and grounded in the book?"**

---

## What's inside

| File | What it does |
|---|---|
| `eval_set.py` | 5 evaluation questions. Every reference answer is written **only** from `data/processed/chunks.jsonl`, with the gold chunk IDs and an exact quote. `python evaluation/eval_set.py` checks that the quotes still match the chunks. |
| `run_deepeval.py` | Runs the full pipeline (hybrid search → rerank → LLM) for each question, scores the answers with DeepEval, and saves one JSON file per model. |
| `test_graceful_failure.py` | Forces the LLM to return invalid output and checks that the pipeline retries, then fails gracefully instead of crashing. |
| `results/` | One `<model>.json` per evaluated model (`:` becomes `_` in filenames, e.g. `llama3.2_3b.json`). |

---

## Quick start

```bash
pip install deepeval                                   # not in requirements.txt (dev-only)

python evaluation/eval_set.py                          # sanity check: questions ↔ chunks
python evaluation/run_deepeval.py                      # default model from app/config.py
python evaluation/run_deepeval.py --model gemma3:4b    # any installed Ollama model
python evaluation/run_deepeval.py --compare            # side-by-side table of all results
```

Before doing any work, the script checks that the eval set matches the chunks and that both the answering model and the judge are installed in Ollama.

---

## Two models, two roles

| Flag | Role | Default |
|---|---|---|
| `--model` | **Generator:** answers the questions (the thing you're comparing) | `LOCAL_MODEL` from `app/config.py` |
| `--judge` | **Evaluator:** scores the answers | `qwen3:4b-instruct` |

💡 **Keep the judge the same across runs.** If the judge changes too, you can't tell whether a score moved because the answer improved or because the grader got stricter or looser.
And if you evaluate the judge model itself as a generator, remember it may be a little generous to its own style of answer.

---

## The metrics

| Metric | Question it answers | Mostly depends on |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved chunks? (a hallucination check) | the LLM |
| **Answer relevancy** | Does the answer actually address the question? | the LLM |
| **Contextual precision** | Are the relevant chunks ranked above the irrelevant ones? | retrieval + reranker |
| **Contextual recall** | Does the retrieved context contain everything in the reference answer? | retrieval + reranker |
| `retrieval_hit` | Did the final 4 chunks include at least one gold chunk? (no LLM needed) | retrieval + reranker |

A handy rule of thumb: if contextual precision and recall are the same for two models, both got the same chunks, so any difference in faithfulness or relevancy comes from the LLM alone.

---

## The eval set

| # | Question | Gold chunks |
|---|---|---|
| e1 | What is machine learning? | 9, 0 |
| e2 | What is concept learning from labeled training examples? | 65, 64 |
| e3 | How can unsupervised clustering be used to place the kernel centers of a radial basis function network? | 696 |
| e4 | What is reinforcement learning? | 1093, 1094, 1143 |
| e5 | What are the main applications of machine learning? | 0, 9, 10, 11 |

**Why e2 and e3 are worded that way:** the book never uses the term *"supervised learning"*, and it only mentions unsupervised learning when talking about clustering. A reference answer that isn't in the book would penalise the pipeline for correctly saying "I don't know". Every reference here can be traced back to real chunks.

---

## What a result file contains

```jsonc
{
  "model": "llama3.2:3b",
  "judge": "qwen3:4b-instruct",
  "timestamp": "2026-09-24T20:36:35",
  "mean_scores": { "faithfulness": 0.903, "answer_relevancy": 0.925, ... },
  "results": [
    {
      "qid": "e1",
      "user_input": "What is machine learning?",
      "response": "...",                        // the model's answer
      "reference": "...",                       // expected answer
      "retrieved_chunk_ids": [0, 48, 41, 49],
      "relevant_chunk_ids": [9, 0],
      "retrieval_hit": true,
      "scores":  { "faithfulness": 0.86, ... },
      "reasons": { "faithfulness": "The judge's explanation...", ... },
      "latency_ms": 3828.5,
      "llm": { "prompt_tokens": 950, "completion_tokens": 60, ... }
    }
  ]
}
```

The `reasons` field is the most useful part when a score looks off: it tells you *which* claim the judge considered unsupported.

---

## Current results

| Model | Faithfulness | Answer relevancy | Contextual precision | Contextual recall | Retrieval hit |
|---|---|---|---|---|---|
| **llama3.2:3b** | **0.90** | 0.93 | 0.98 | 0.89 | 5/5 |
| gemma3:4b | 0.76 | 0.92 | 0.98 | 0.89 | 5/5 |

Judge: `qwen3:4b-instruct`. With only 5 questions, a single answer moves the averages noticeably, so treat small gaps as a hint rather than a verdict.

---

## Tips

- **Changed the chunks?** Run `python evaluation/eval_set.py` first. If a quote no longer matches, the chunk IDs have shifted.
- **Adding questions:** copy a row in `EVAL_SET`, find the answer in `chunks.jsonl`, and paste an exact quote into `evidence`. The validator keeps you honest.
- **`test_graceful_failure.py`** needs Ollama running, but it mocks the LLM output, so it's fast.
