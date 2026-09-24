# benchmarks/

1. **Which retrieval method finds the right chunks?** Use `bench_retrieval.py`. 
2. **Which local model is fast enough on my GPU?** Use `bench_llm_speed.py`.

Run everything from the repo root, in the same env you use for the app.

## Files

| File | What it is |
|---|---|
| `golden_set.jsonl` | 69 hand-labelled questions spread over all 13 chapters. Each row has `relevant_chunk_ids`, a verbatim `evidence` quote from the primary chunk, a `reference_answer` (for DeepEval later) and a `qtype`. |
| `validate_golden.py` | Checks that the labels still match `data/processed/chunks.jsonl`. **Re-run after any re-chunking**, because chunk ids shift. |
| `retrievers.py` | One `search(query, k) -> [chunk_id]` function per method. |
| `bench_retrieval.py` | Reports Hit@k, Recall@k, MRR, nDCG and latency per method, plus per-question CSVs and a list of misses. |
| `bench_llm_speed.py` | Reports TTFT, prefill tok/s, decode tok/s, total latency, token counts and VRAM per model. |
| `bench_config.py` | All the settings (methods, models, k values, num_ctx...). |
| `cache/` | Saved retrieval rankings (git-ignored). |
| `results/` | Timestamped CSVs from every run. |

## Retrieval

### How it works

```
golden question ──► method searches all chunks in chunks.jsonl ──► ranked chunk ids
                                                                        │
golden relevant_chunk_ids ◄────────────── compare (Hit/MRR/nDCG) ───────┘
```

The golden set is the exam (questions + answer key). The chunks are what gets searched, which is the same data the app searches.

### Methods

|---|---|---|
| `bm25` | Keyword search. Uses the same regex tokenizer as `app/retrieval.py` (strips markdown like `**_entropy,_**` → `entropy`). | No |
| `chroma` | Dense search: the query is embedded with `bge-small` and compared with the stored chunk vectors. `vectorstore.as_retriever("similarity")` is only a LangChain wrapper around this same search, so it isn't a separate method. | Embedding model (query only; chunk vectors are already in `data/chroma`) |
| `chroma_mmr` | Optional. Dense search that also penalises near-duplicate results (Maximal Marginal Relevance). | Embedding model |
| `hybrid` | `app/retrieval.hybrid_search`: BM25 top-20 + dense top-20 fused with Reciprocal Rank Fusion. | Embedding model |
| `rerank` | Hybrid top-20 re-scored by the cross-encoder in `app/reranker.py`. It reads the question and each chunk together, so it can't be precomputed. | Embedding model + cross-encoder |

### Run

```bash
python benchmarks/validate_golden.py
python benchmarks/bench_retrieval.py                              # methods from bench_config.ENABLED_METHODS
python benchmarks/bench_retrieval.py --methods bm25,chroma        # any subset
python benchmarks/bench_retrieval.py --methods chroma,chroma_mmr
python benchmarks/bench_retrieval.py --refresh                    # ignore the cache
```

A method's code (and its models) is only loaded if you list it, so `bm25` works before Chroma exists.

### Cache: models only load when something changed

Each method's rankings are saved to `cache/<method>.json`, with a fingerprint of `chunks.jsonl`, `app/config.py`, `app/retrieval.py`, `app/reranker.py` and `retrievers.py`. If none of them changed, the next run reads the cache and loads **no models**. If one did change (new chunks, new embedding model, edited retrieval code), that method is recomputed automatically. The latency shown comes from the run that built the cache.

### Metrics

| Metric | Meaning |
|---|---|
| Hit@k | 1 if any relevant chunk is in the top k. "Did we find it at all?" |
| Recall@k | Fraction of the relevant chunks found in the top k. Matters when the answer is spread over two chunks. |
| MRR@k | 1 / rank of the first relevant chunk. "How high did it rank?" |
| nDCG@k | Like MRR, but gives credit to every relevant chunk by position. |

## LLM speed

```bash
ollama pull gemma3:4b && ollama pull llama3.2:3b && ollama pull phi4-mini:3.8b && ollama pull granite4:micro
python benchmarks/bench_llm_speed.py
python benchmarks/bench_llm_speed.py --models granite4:micro --repeats 2
```

- Models run **one at a time**: warm-up → prompts → unload. Two 4B models on 6 GB would fight for VRAM and ruin the timings.
- Every model gets the same RAG prompts (a golden question + 4 context chunks), with `temperature=0`, the same `seed`, `num_ctx` and `num_predict`.
- **Prefill** = reading the ~930-token prompt (fast, parallel, sets TTFT). **Decode** = writing the answer one token at a time (the usual "tok/s").
- Each request starts with a random `[run xxxx]` tag. Without it, Ollama reuses the KV cache from an identical earlier prompt, skips prefill, and TTFT / prefill tok/s come out about 10x too good.
- `thinking_chars > 0` means a reasoning model slipped in. Its speed isn't comparable to instruct models (set `"think": False` for it in `bench_config.py`).
- If `vram_gb` is lower than `size_gb`, the model spilled out of the 6 GB GPU and isn't a fair comparison.
- `total_ms` depends on answer length. A model that writes twice as much looks slower, so compare `decode_tps` and `ttft_ms` too.

## Picking the winner

Speed alone doesn't pick the model. Once you have the top 2-3 retrieval configs and models, run DeepEval faithfulness and answer relevancy using the `reference_answer` fields (the answers are already saved in `results/llm_speed_runs_*.csv`). Then plot quality against `total_ms_p50`: the best-value options are the ones with high quality and low latency.
