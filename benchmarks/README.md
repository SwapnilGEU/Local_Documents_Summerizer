# Benchmarks

This folder contains the benchmarks used to select the retrieval strategy and the LLM for PDFinsight. It covers two independent questions:

1. **Retrieval accuracy:** which search method ranks the correct chunk highest (`bench_retrieval.py`)
2. **LLM speed:** which local model responds fastest on the target GPU (`bench_llm_speed.py`)

They are kept separate because retrieval quality does not depend on the LLM, so it can be tuned without generating any text. Answer quality is measured separately in [`evaluation/`](../evaluation/README.md).

## Modules

| Module | Purpose |
|---|---|
| `bench_config.py` | Central configuration: retrieval methods, k values, candidate models, generation options and repeat counts. |
| `golden_set.jsonl` | 69 hand-written questions covering all 13 chapters. Each entry records the relevant chunk IDs, a verbatim evidence quote, a reference answer and a question type. |
| `validate_golden.py` | Confirms that every evidence quote is still present in its labelled chunk. Re-chunking shifts chunk IDs, and this check detects stale labels. |
| `retrievers.py` | One `search(question, k) -> [chunk_id]` function per method, behind a common interface. Adding a method requires a single function. |
| `bench_retrieval.py` | Runs each method against the golden set, computes ranking metrics, caches rankings and writes CSV reports. |
| `bench_llm_speed.py` | Sends identical RAG prompts to each Ollama model and records latency, throughput, token counts and VRAM usage. |
| `cache/` | Cached rankings per method (git-ignored). |
| `results/` | Timestamped CSV reports for every run. |

## Retrieval Benchmark

Each golden question is run against all 1,234 chunks in `data/processed/chunks.jsonl`, which is the same data the application searches. The ranked chunk IDs returned are compared with the labelled relevant chunks.

### Methods

| Method | Description | Models loaded |
|---|---|---|
| `bm25` | Keyword search. Uses the same regex tokenizer as `app/retrieval.py`, which strips markdown artifacts. | None |
| `chroma` | Dense vector search with `bge-small-en-v1.5`. `as_retriever("similarity")` wraps this same call. | Embedding model |
| `chroma_mmr` | Optional. Dense search with Maximal Marginal Relevance to reduce near-duplicate results. | Embedding model |
| `hybrid` | BM25 and dense top-20 results fused with Reciprocal Rank Fusion (`app/retrieval.hybrid_search`). | Embedding model |
| `rerank` | Hybrid top-20 results re-scored by the cross-encoder in `app/reranker.py`. | Embedding model, cross-encoder |

A method is loaded only when it is requested, so `bm25` runs without a vector store.

### Usage

```bash
python benchmarks/validate_golden.py
python benchmarks/bench_retrieval.py                          # methods from ENABLED_METHODS
python benchmarks/bench_retrieval.py --methods bm25,rerank    # a specific subset
python benchmarks/bench_retrieval.py --refresh                # ignore the cache
```

The output includes a summary table, Hit@5 by question type, and a list of missed questions showing where the correct chunk actually ranked.

### Metrics

| Metric | Definition |
|---|---|
| Hit@k | 1 if any relevant chunk appears in the top k |
| Recall@k | Fraction of relevant chunks found in the top k |
| MRR@k | Reciprocal rank of the first relevant chunk |
| nDCG@k | Position-weighted credit for every relevant chunk |

Rankings are de-duplicated before scoring, so a retriever that returns the same chunk twice cannot be credited twice.

### Caching

Rankings are stored in `cache/<method>.json` together with a fingerprint of `chunks.jsonl`, `app/config.py`, `chunking.py`, `vectorstore.py`, `retrieval.py`, `reranker.py` and `retrievers.py`. When none of these have changed, a re-run loads no models. When any of them change, the affected method is recomputed automatically. Reported latencies come from the run that built the cache.

### Results

| Method | Hit@1 | Hit@5 | Recall@8 | MRR@8 | Latency (p50) |
|---|---|---|---|---|---|
| bm25 | 0.59 | 0.93 | 0.89 | 0.72 | 4 ms |
| hybrid | 0.62 | 0.96 | 0.94 | 0.75 | 36 ms |
| **rerank** | **0.70** | **0.97** | **0.95** | **0.81** | 129 ms |

Dense-only results will be added after the next run, following the vector store fix below.

### Findings

- **Tokenization:** BM25 with whitespace tokenization reached Hit@5 = 0.80. A regex tokenizer that strips markdown raised it to 0.93, with no other change.
- **Vector store integrity:** early dense runs showed Hit@1 equal to Hit@5, because every chunk was stored five times in Chroma. `app/vectorstore.py` now uses deterministic IDs and rebuilds the collection when it is out of sync.

## LLM Speed Benchmark

### Usage

```bash
for m in llama3.2:3b gemma3:4b phi4-mini:3.8b granite4:micro qwen3:4b-instruct; do ollama pull $m; done   # bash; pull one at a time in cmd
python benchmarks/bench_llm_speed.py
python benchmarks/bench_llm_speed.py --models granite4:micro --repeats 2
```

### Methodology

- **Isolation:** models are benchmarked one at a time (warm-up, measured runs, unload), so they never compete for VRAM.
- **Identical inputs:** every model receives the same 10 prompts, each a golden question with 4 context chunks (~930 tokens). Temperature, seed, `num_ctx` and `num_predict` are fixed. Each prompt is repeated 3 times and medians are reported.
- **Cache busting:** each request begins with a unique tag. Without it, Ollama reuses the KV cache for repeated prompts, skips prefill, and understates time to first token by roughly 10×.
- **Reasoning detection:** `thinking_chars` records any reasoning tokens, so that reasoning models are not compared directly with instruct models.

### Columns

| Column | Definition |
|---|---|
| `ttft_ms` | Time to first token, dominated by prompt prefill in RAG workloads |
| `prefill_tps` | Prompt tokens processed per second |
| `decode_tps` | Output tokens generated per second |
| `total_ms` | End-to-end generation time. Depends on answer length, so read it alongside `gen_tokens`. |
| `vram_gb` / `size_gb` | VRAM lower than model size indicates a partial offload to system RAM |

### Results (RTX 4050, 6 GB)

| Model | VRAM | TTFT | Decode (tok/s) | Total | Output tokens |
|---|---|---|---|---|---|
| **llama3.2:3b** | 2.6 GB | **290 ms** | **72.9** | **0.94 s** | 43 |
| granite4:micro | 2.5 GB | 370 ms | 65.1 | 1.47 s | 72 |
| phi4-mini:3.8b | 3.1 GB | 389 ms | 59.5 | 1.80 s | 78 |
| qwen3:4b-instruct | 3.2 GB | 463 ms | 55.8 | 2.35 s | 110 |
| gemma3:4b | 2.9 GB | 528 ms | 55.0 | 1.29 s | 40 |

Generated answers are saved in `results/llm_speed_runs_*.csv` for later quality evaluation.

## Selection Process

1. Choose the retrieval configuration with `bench_retrieval.py` (hybrid + reranker).
2. Shortlist models that meet latency targets with `bench_llm_speed.py`.
3. Score answer quality for the shortlisted models with `evaluation/run_deepeval.py`.
4. Select the model with the best quality at an acceptable latency (`llama3.2:3b`).
