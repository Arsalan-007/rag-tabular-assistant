# Evaluation results

- Corpus: **41 papers**, BAAI/bge-small-en-v1.5 embeddings, ChromaDB.
- Eval set: **40 labeled questions** (`src/eval_questions.py`), paper-level ground truth.
- Hardware: CPU only.

## Method

- **hit-rate@k** — fraction of questions where at least one gold paper appears among the papers of the top-k retrieved *chunks*.
- **MRR** — mean of 1 / (rank of the first chunk whose paper is a gold paper).
- **guard** — of the out-of-domain probes, how many correctly trip the low-confidence flag (best dense cosine < 0.6).
- Latency is end-to-end retrieval per query (embed + search + fuse + rerank), steady-state, CPU.

## Retrieval ablation

| pipeline | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms | p95 ms | guard |
|---|---|---|---|---|---|---|---|---|
| dense | 0.825 | 0.900 | 0.950 | 0.975 | 0.878 | 19 | 26 | 4/4 |
| hybrid (BM25 + RRF) | 0.825 | 0.925 | 0.925 | 0.975 | 0.879 | 42 | 55 | 4/4 |
| dense + rerank | 0.800 | 0.950 | 0.975 | 1.000 | 0.880 | 2237 | 3037 | 4/4 |
| hybrid + rerank | 0.775 | 0.925 | 0.950 | 1.000 | 0.859 | 2282 | 3314 | 4/4 |
| hybrid + rerank + rewrite | 0.850 | 0.950 | 0.950 | 0.975 | 0.899 | 4555 | 6048 | 4/4 |

## Takeaways

- **Reranking is the decisive stage.** It lifts hit@10 to 1.000 and hit@3 to 0.950, for ~2237 ms/query median (vs ~19 ms without it) — the cross-encoder dominates latency.
- **BM25 helps on its own**: hybrid beats dense at MRR (0.879 vs 0.878) and hit@1 (0.825 vs 0.825) — exact-term matches (method acronyms, symbols) that a 33M-param embedder blurs.
- **Under reranking, hybrid vs dense is within noise** on this 40-question set (~1 question per point). The eval over-samples clean "how does X work" questions; hybrid is kept on by default as a safety net for rarer exact-term queries.
- **Guard rail**: 4/4 out-of-domain probes correctly flagged low-confidence in every configuration.

Configured default: `bge-small → BM25 → RRF(k=60) → rerank[bge-reranker-base]  (fetch_k=20, top_k=5)`.

## Remaining misses (gold paper not in top-10)

**dense** — 1 miss(es):
- _Is deep learning all you need for tabular data, or do gradient-boosted trees still win on typical benchmarks?_  
  wanted `['2106.03253']`, got `['2110.01889', '2410.24210', '2106.11189', '2207.08815', '1909.06312']`

**hybrid (BM25 + RRF)** — 1 miss(es):
- _How well do simple, well-regularised MLPs perform on tabular benchmarks when their regularisation is tuned?_  
  wanted `['2106.11189']`, got `['2410.24210', '2604.15297', '2203.05556', '2106.01342', '2207.03208']`

**hybrid + rerank + rewrite** — 1 miss(es):
- _Is deep learning all you need for tabular data, or do gradient-boosted trees still win on typical benchmarks?_  
  wanted `['2106.03253']`, got `['2110.01889', '2410.24210', '2106.11189', '1909.06312', '2207.08815']`


## Answer faithfulness (LLM-as-judge)

Mean faithfulness **5.00 / 5** over 40/40 gradable answers (judge: gemini · gemini-flash-lite-latest).

Reported apart from retrieval, and read with caution:

- The generator is told to answer *only* from the provided passages and to say so when they're insufficient. A near-ceiling score mostly confirms that instruction is being followed — it is **not** an adversarial test.
- The judge is the same model family as the generator, which biases it toward leniency.
- A discriminating version would inject known-unsupported claims and check the judge catches them, and/or use a stronger, different judge model. That's future work.

---

_Regenerate: `python src/evaluate.py --ablation --out eval/results.md`_
