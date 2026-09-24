# 📏 benchmarks/

Measure before you choose. This folder answers two separate questions about the PDFinsight pipeline:

1. **Retrieval:** which search method puts the right chunk in front of the LLM? → `bench_retrieval.py`
2. **LLM speed:** which local model is fast enough on this GPU? → `bench_llm_speed.py`

They're kept apart on purpose. Retrieval quality doesn't depend on the LLM, so it can be tuned quickly without generating any text. Speed is then compared on identical prompts.
Answer *quality* lives in [`../evaluation`](../evaluation/README.md).

---

## 🧩 What each module does

| Module | Role | Why it exists |
|---|---|---|
| `bench_config.py` | Every setting in one place: methods, k values, models, `num_ctx`, repeats. | Change an experiment without touching code. |
| `golden_set.jsonl` | 69 hand-written questions over all 13 chapters. Each has `relevant_chunk_ids`, an exact `evidence` quote, a `reference_answer` and a `qtype`. | The **answer key**. Without ground truth, "retrieval accuracy" can't be measured. |
| `validate_golden.py` | Checks that every evidence quote is still inside its chunk. | Re-chunking shifts chunk IDs. This catches silently wrong labels. |
| `retrievers.py` | One `search(question, k) -> [chunk_id]` function per method. | A common interface: the benchmark doesn't care *how* a method works, only what it returns. Adding a method = one function. |
| `bench_retrieval.py` | Runs each method over the golden set, scores the rankings, caches them and writes CSVs. | Compares search methods fairly on the same questions. |
| `bench_llm_speed.py` | Sends the same RAG prompts to each Ollama model and records timings, tokens and VRAM. | Compares models on *your* hardware with *your* prompt sizes. |
| `cache/` | Saved rankings per method (git-ignored). | Re-runs skip loading the embedding model and reranker when nothing changed. |
| `results/` | Timestamped CSVs (summary + per-question / per-run). | A history of every experiment. |

---

## 🔎 Retrieval benchmark

### How it works

```
golden question ──► method searches all 1,234 chunks ──► ranked chunk IDs
                                                             │
golden relevant_chunk_ids ◄──────── compare ─────────────────┘
                       → Hit@k, Recall@k, MRR, nDCG, latency
```

The golden set is the exam (questions + answer key). The **chunks** are what gets searched, which is the same data the app searches.

### Methods

| Method | What it does | Loads a model? |
|---|---|---|
| `bm25` | Keyword search. Regex tokenizer, the same one as `app/retrieval.py`, strips markdown like `**_entropy,_**` → `entropy`. | No |
| `chroma` | Dense search: embeds the question with `bge-small` and finds the nearest stored chunk vectors. (`as_retriever("similarity")` is just a LangChain wrapper around this same call.) | Embedding model |
| `chroma_mmr` | *Optional.* Dense search that also avoids near-duplicate results (Maximal Marginal Relevance). | Embedding model |
| `hybrid` | `app/retrieval.hybrid_search`: BM25 top-20 + dense top-20 fused with Reciprocal Rank Fusion. | Embedding model |
| `rerank` | Hybrid top-20 re-scored by the cross-encoder in `app/reranker.py`. | Embedding model + cross-encoder |

Methods are loaded only when listed, so `bm25` runs even before a vector store exists.

### Run it

```bash
python benchmarks/validate_golden.py
python benchmarks/bench_retrieval.py                         # methods from ENABLED_METHODS
python benchmarks/bench_retrieval.py --methods bm25,rerank    # any subset
python benchmarks/bench_retrieval.py --refresh               # ignore the cache
```

You'll get a summary table, a breakdown by question type, and the questions each method missed, with where the right chunk actually ranked. That list is where most insights come from.

### Metrics

| Metric | Meaning |
|---|---|
| **Hit@k** | 1 if *any* relevant chunk is in the top k. "Did we find it at all?" |
| **Recall@k** | Share of the relevant chunks found in the top k. Matters when an answer spans two chunks. |
| **MRR@k** | 1 / rank of the first relevant chunk. "How high did it rank?" |
| **nDCG@k** | Like MRR, but credits every relevant chunk by position. |

### The cache (why models don't always load)

Rankings are saved to `cache/<method>.json` together with a fingerprint of `chunks.jsonl`, `app/config.py`, `chunking.py`, `vectorstore.py`, `retrieval.py`, `reranker.py` and `retrievers.py`.
If none of them changed, the next run loads **no models** and finishes in seconds. If any of them changed, that method is recomputed automatically. The latency shown is from the run that built the cache.

### Latest results (69 questions)

| Method | Hit@1 | Hit@5 | Recall@8 | MRR@8 | Latency p50 |
|---|---|---|---|---|---|
| bm25 | 0.59 | 0.93 | 0.89 | 0.72 | 4 ms |
| chroma | *re-run pending* | | | | 32 ms |
| hybrid | 0.62 | 0.96 | 0.94 | 0.75 | 36 ms |
| **rerank** | **0.70** | **0.97** | **0.95** | **0.81** | 129 ms |

Lessons learned along the way:

- **Tokenization matters:** BM25 with `.split()` scored Hit@5 = 0.80. With the regex tokenizer it scored 0.93, with no other change.
- **Check your data:** the first dense runs had Hit@1 = Hit@5, because every chunk was stored in Chroma 5 times. `app/vectorstore.py` now detects and repairs that.

---

## ⚡ LLM speed benchmark

### Run it

```bash
ollama pull gemma3:4b && ollama pull llama3.2:3b && ollama pull phi4-mini:3.8b && ollama pull granite4:micro
python benchmarks/bench_llm_speed.py
python benchmarks/bench_llm_speed.py --models granite4:micro --repeats 2
```

### How it keeps the comparison fair

- **One model at a time:** warm-up → prompts → unload. Two 4B models on 6 GB would compete for VRAM and distort the timings.
- **Identical inputs:** the same golden question + 4 context chunks (~930 tokens) for every model, with `temperature=0`, a fixed `seed`, `num_ctx` and `num_predict`.
- **No prompt-cache shortcuts:** each request starts with a random `[run xxxx]` tag. Without it, Ollama reuses its KV cache for a repeated prompt, skips prefill, and the time to first token looks ~10× too good.
- **Reasoning models flagged:** `thinking_chars > 0` means the model spent tokens "thinking", so its numbers aren't comparable to instruct models.

### What the columns mean

| Column | Meaning |
|---|---|
| `ttft_ms` | Time to first token. In RAG this is mostly **prefill** (reading the prompt). |
| `prefill_tps` | Prompt tokens processed per second (fast and parallel). |
| `decode_tps` | Answer tokens generated per second (one at a time), the usual "tok/s". |
| `total_ms` | Full request time. Depends on answer length, so read it next to `gen_tokens`. |
| `vram_gb` vs `size_gb` | If VRAM is lower, the model spilled into system RAM and runs slower. |

### Latest results (RTX 4050, 6 GB)

| Model | VRAM | TTFT | Decode tok/s | Total | Answer length |
|---|---|---|---|---|---|
| **llama3.2:3b** | 2.6 GB | **290 ms** | **72.9** | **0.94 s** | 43 tok |
| granite4:micro | 2.5 GB | 370 ms | 65.1 | 1.47 s | 72 tok |
| phi4-mini:3.8b | 3.1 GB | 389 ms | 59.5 | 1.80 s | 78 tok |
| qwen3:4b-instruct | 3.2 GB | 463 ms | 55.8 | 2.35 s | 110 tok |
| gemma3:4b | 2.9 GB | 528 ms | 55.0 | 1.29 s | 40 tok |

Answers from every run are saved in `results/llm_speed_runs_*.csv`, ready for a quality check.

---

## 🧭 From numbers to a decision

Speed alone doesn't pick a model. The workflow used for this project:

1. `bench_retrieval.py` → pick the retrieval setup (hybrid + reranker).
2. `bench_llm_speed.py` → shortlist models that are fast on the GPU.
3. `../evaluation/run_deepeval.py --model <name>` → check faithfulness and relevancy.
4. Choose the best quality at a latency you're happy with. For this project that's `llama3.2:3b`.
