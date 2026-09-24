# Evaluation

End-to-end answer quality evaluation for PDFinsight using [DeepEval](https://github.com/confident-ai/deepeval). While [`benchmarks/`](../benchmarks/README.md) measures retrieval accuracy and generation speed, this folder measures whether the final answers are correct and grounded in the source document.

## Modules

| Module | Purpose |
|---|---|
| `eval_set.py` | Five evaluation questions. Each has a reference answer written only from `data/processed/chunks.jsonl`, the relevant chunk IDs and a verbatim evidence quote. Running the file checks that the quotes still match the chunks. |
| `run_deepeval.py` | Runs the full pipeline (hybrid search, reranking, generation) for each question, scores the answers with DeepEval and saves one JSON file per model. |
| `test_graceful_failure.py` | Forces invalid LLM output to confirm that validation retries run and the pipeline fails gracefully. |
| `results/` | One result file per evaluated model, e.g. `llama3.2_3b.json`. Colons in model names are replaced for Windows compatibility. |

## Usage

```bash
pip install deepeval                                   # development dependency

python evaluation/eval_set.py                          # validate questions against chunks
python evaluation/run_deepeval.py                      # model from app/config.py
python evaluation/run_deepeval.py --model gemma3:4b    # any installed Ollama model
python evaluation/run_deepeval.py --compare            # compare all saved results
```

Before running, the script checks the evaluation set and confirms that both the generator and the judge model are installed in Ollama.

### Generator and judge

| Option | Role | Default |
|---|---|---|
| `--model` | Generates the answers being evaluated | `LOCAL_MODEL` in `app/config.py` |
| `--judge` | Scores the answers | `qwen3:4b-instruct` |

The judge is held constant across runs so that score differences reflect the generator rather than changes in grading. Evaluating the judge model as a generator may introduce some self-preference bias.

## Metrics

| Metric | Measures | Primarily reflects |
|---|---|---|
| Faithfulness | Whether every claim in the answer is supported by the retrieved context | Generator |
| Answer relevancy | Whether the answer addresses the question | Generator |
| Contextual precision | Whether relevant chunks are ranked above irrelevant ones | Retrieval |
| Contextual recall | Whether the retrieved context covers the reference answer | Retrieval |
| `retrieval_hit` | Whether the final four chunks include a labelled relevant chunk | Retrieval |

When two models share identical contextual precision and recall, they received the same context, so differences in faithfulness and relevancy can be attributed to the generator.

## Evaluation Set

| ID | Question | Relevant chunks |
|---|---|---|
| e1 | What is machine learning? | 9, 0 |
| e2 | What is concept learning from labeled training examples? | 65, 64 |
| e3 | How can unsupervised clustering be used to place the kernel centers of a radial basis function network? | 696 |
| e4 | What is reinforcement learning? | 1093, 1094, 1143 |
| e5 | What are the main applications of machine learning? | 0, 9, 10, 11 |

Questions e2 and e3 are phrased around the source's own terminology. The textbook does not use the term "supervised learning", and it discusses unsupervised learning only in the context of clustering. A reference answer containing content absent from the corpus would penalise the pipeline for correctly declining to answer.

## Output Format

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
      "response": "...",
      "reference": "...",
      "retrieved_chunk_ids": [0, 48, 41, 49],
      "relevant_chunk_ids": [9, 0],
      "retrieval_hit": true,
      "scores":  { "faithfulness": 0.86, ... },
      "reasons": { "faithfulness": "...", ... },
      "latency_ms": 3828.5,
      "llm": { "prompt_tokens": 950, "completion_tokens": 60, ... }
    }
  ]
}
```

The `reasons` field contains the judge's explanation for each score and is the first place to look when a score is unexpected.

## Results

| Model | Faithfulness | Answer relevancy | Contextual precision | Contextual recall | Retrieval hit |
|---|---|---|---|---|---|
| **llama3.2:3b** | **0.90** | 0.93 | 0.98 | 0.89 | 5/5 |
| gemma3:4b | 0.76 | 0.92 | 0.98 | 0.89 | 5/5 |

Judge: `qwen3:4b-instruct`. With five questions, a single answer can shift the averages noticeably, so small differences should be read as indicative rather than conclusive.

## Maintenance

- After re-chunking, run `python evaluation/eval_set.py`. A mismatch means the chunk IDs have shifted.
- To add a question, add an entry to `EVAL_SET` with an exact quote from `chunks.jsonl` in `evidence`, and the validator will confirm it.
- `test_graceful_failure.py` mocks the LLM output but still requires Ollama to be running, because the pipeline checks the model at import time.
