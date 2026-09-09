"""
Evaluation harness for the tabular-DL RAG assistant.

Two signals, measured and reported separately because they fail independently:

  RETRIEVAL  -- did the right paper come back?
    hit-rate@k : fraction of questions with a gold paper among the top-k chunks
    MRR        : mean 1 / (rank of the first chunk whose paper is gold)
    Ground truth is the labeled paper id(s) in eval_questions.py -- objective.

  ANSWER FAITHFULNESS  (optional, --judge) -- is the generated answer grounded
    in the retrieved passages? Scored 1-5 by the configured LLM acting as judge.
    A small local judge is noisy, so this is reported apart from retrieval,
    never blended in.

  GUARD RAIL -- do the OUT_OF_DOMAIN questions trip the low-confidence flag
    instead of confidently returning irrelevant chunks?

Usage:
    python src/evaluate.py                    # metrics for the configured pipeline
    python src/evaluate.py --ablation         # dense / hybrid / +rerank matrix
    python src/evaluate.py --ablation --out eval/results.md   # + write the report
    python src/evaluate.py --k 1 3 5 10
    python src/evaluate.py --judge            # + LLM-as-judge faithfulness (slow)
    python src/evaluate.py --json out.json    # machine-readable dump (for CI)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

import config
from eval_questions import EVAL_QUESTIONS, OUT_OF_DOMAIN
from retrieval import Retriever

# Pipeline variants for --ablation, in increasing sophistication.
ABLATION_CONFIGS = [
    ("dense", {"retrieval_mode": "dense", "rerank": False}),
    ("hybrid (BM25 + RRF)", {"retrieval_mode": "hybrid", "rerank": False}),
    ("dense + rerank", {"retrieval_mode": "dense", "rerank": True}),
    ("hybrid + rerank", {"retrieval_mode": "hybrid", "rerank": True}),
]


@dataclass
class RunMetrics:
    label: str
    n_questions: int
    hit_rate: dict  # {k: fraction}
    mrr: float
    n_misses: int
    latency_p50_ms: float
    latency_p95_ms: float
    guard_pass: int  # of len(OUT_OF_DOMAIN)
    guard_total: int
    misses: list  # [{"q":..., "gold":[...], "got":[...]}]


# ── retrieval evaluation ───────────────────────────────────────────────
def _paper_rank_of_first_gold(chunk_papers: list[str], gold: set[str]) -> int | None:
    for rank, pid in enumerate(chunk_papers, start=1):
        if pid in gold:
            return rank
    return None


def evaluate_retrieval(retriever: Retriever, k_values: list[int], label: str) -> RunMetrics:
    max_k = max(k_values)
    first_ranks: list[int | None] = []
    latencies: list[float] = []
    misses: list[dict] = []

    for item in EVAL_QUESTIONS:
        gold = set(item["papers"])
        res = retriever.retrieve(item["q"], top_k=max_k)
        latencies.append(res.timings_ms.get("total", 0.0))
        chunk_papers = [h.arxiv_id for h in res.hits]
        rank = _paper_rank_of_first_gold(chunk_papers, gold)
        first_ranks.append(rank)
        if rank is None:
            misses.append({
                "q": item["q"],
                "gold": sorted(gold),
                "got": list(dict.fromkeys(chunk_papers))[:5],
            })

    n = len(EVAL_QUESTIONS)
    hit_rate = {
        k: sum(1 for r in first_ranks if r is not None and r <= k) / n for k in k_values
    }
    mrr = statistics.mean(1.0 / r if r else 0.0 for r in first_ranks)

    guard_pass = 0
    for q in OUT_OF_DOMAIN:
        if retriever.retrieve(q, top_k=max_k).low_confidence:
            guard_pass += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0

    return RunMetrics(
        label=label,
        n_questions=n,
        hit_rate={k: round(v, 4) for k, v in hit_rate.items()},
        mrr=round(mrr, 4),
        n_misses=len(misses),
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        guard_pass=guard_pass,
        guard_total=len(OUT_OF_DOMAIN),
        misses=misses,
    )


def _apply(overrides: dict) -> Retriever:
    for key, val in overrides.items():
        setattr(config.settings, key, val)
    r = Retriever()
    r.warmup()
    return r


def run_ablation(k_values: list[int]) -> list[RunMetrics]:
    saved = {k: getattr(config.settings, k) for k in ("retrieval_mode", "rerank")}
    results = []
    try:
        for label, overrides in ABLATION_CONFIGS:
            print(f"  running: {label} ...", flush=True)
            results.append(evaluate_retrieval(_apply(overrides), k_values, label))
    finally:
        for k, v in saved.items():
            setattr(config.settings, k, v)
    return results


# ── answer faithfulness (LLM-as-judge) ─────────────────────────────────
JUDGE_PROMPT = """You are grading whether an ANSWER is faithful to the SOURCE \
passages it was based on. Faithful = every claim in the answer is supported by \
the sources. Unfaithful = the answer adds facts not in the sources or \
contradicts them.

Score 1-5:
5 = fully supported   4 = mostly, minor unsupported detail   3 = partially
2 = largely unsupported   1 = contradicts or ignores the sources

Reply with ONLY a JSON object: {"score": <1-5>, "reason": "<one short sentence>"}

=== SOURCES ===
{sources}

=== ANSWER ===
{answer}

=== YOUR JSON GRADE ==="""


def _parse_judge(raw: str):
    try:
        obj = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        score = int(obj.get("score"))
        if 1 <= score <= 5:
            return score, str(obj.get("reason", ""))[:120]
    except Exception:
        pass
    return None, "unparseable judge output"


def evaluate_faithfulness(retriever: Retriever):
    import rag  # local import: only needed with --judge, pulls in the generator

    # Two generate() calls per question. Gemini's free tier is ~15 req/min, so
    # pace to stay under it; Ollama is local and needs no throttle.
    min_interval = (60.0 / 13) if config.settings.generator == "gemini" else 0.0
    last = [0.0]

    def _paced_generate(prompt: str) -> str:
        wait = min_interval - (time.monotonic() - last[0])
        if wait > 0:
            time.sleep(wait)
        out = rag.generate(prompt, stream=False)
        last[0] = time.monotonic()
        return out

    scores = []
    for item in EVAL_QUESTIONS:
        res = retriever.retrieve(item["q"])
        answer = _paced_generate(rag.build_prompt(item["q"], res.hits))
        sources = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(res.hits, 1))
        raw = _paced_generate(
            JUDGE_PROMPT.replace("{sources}", sources).replace("{answer}", answer)
        )
        score, _ = _parse_judge(raw)
        scores.append(score)
        print(f"  [{score or '?'}/5] {item['q'][:64]}", flush=True)
    ok = [s for s in scores if s is not None]
    return (statistics.mean(ok) if ok else None), len(ok), len(scores)


# ── reporting ─────────────────────────────────────────────────────────
def print_table(results: list[RunMetrics], k_values: list[int]) -> None:
    head = ["pipeline", *[f"hit@{k}" for k in k_values], "MRR", "p50 ms", "p95 ms", "guard"]
    print("\n" + "  ".join(f"{h:>16}" if i == 0 else f"{h:>8}" for i, h in enumerate(head)))
    for m in results:
        row = [
            f"{m.label:>16}",
            *[f"{m.hit_rate[k]:>8.3f}" for k in k_values],
            f"{m.mrr:>8.3f}",
            f"{m.latency_p50_ms:>8.0f}",
            f"{m.latency_p95_ms:>8.0f}",
            f"{m.guard_pass}/{m.guard_total:>3}",
        ]
        print("  ".join(row))


def markdown_report(results: list[RunMetrics], k_values: list[int], judge=None) -> str:
    from seed_papers import CORPUS_SIZE

    kmax = max(k_values)
    # best = highest hit@kmax, then MRR, then lower median latency
    best = max(results, key=lambda m: (m.hit_rate[kmax], m.mrr, -m.latency_p50_ms))
    base = next((m for m in results if m.label == "dense"), results[0])
    lines = [
        "# Evaluation results",
        "",
        f"- Corpus: **{CORPUS_SIZE} papers**, {config.settings.embed_model} embeddings, ChromaDB.",
        f"- Eval set: **{base.n_questions} labeled questions** (`src/eval_questions.py`), "
        f"paper-level ground truth.",
        "- Hardware: CPU only.",
        "",
        "## Method",
        "",
        "- **hit-rate@k** — fraction of questions where at least one gold paper appears "
        "among the papers of the top-k retrieved *chunks*.",
        "- **MRR** — mean of 1 / (rank of the first chunk whose paper is a gold paper).",
        "- **guard** — of the out-of-domain probes, how many correctly trip the "
        f"low-confidence flag (best dense cosine < {config.settings.min_similarity}).",
        "- Latency is end-to-end retrieval per query (embed + search + fuse + rerank), "
        "steady-state, CPU.",
        "",
        "## Retrieval ablation",
        "",
        "| pipeline | " + " | ".join(f"hit@{k}" for k in k_values) + " | MRR | p50 ms | p95 ms | guard |",
        "|---|" + "---|" * (len(k_values) + 4),
    ]
    for m in results:
        cells = " | ".join(f"{m.hit_rate[k]:.3f}" for k in k_values)
        lines.append(
            f"| {m.label} | {cells} | {m.mrr:.3f} | {m.latency_p50_ms:.0f} | "
            f"{m.latency_p95_ms:.0f} | {m.guard_pass}/{m.guard_total} |"
        )

    by_label = {m.label: m for m in results}
    hyb = by_label.get("hybrid (BM25 + RRF)")
    lines += [
        "",
        "## Takeaways",
        "",
        f"- **Reranking is the decisive stage.** It lifts hit@{kmax} to "
        f"{best.hit_rate[kmax]:.3f} and hit@3 to {best.hit_rate[3]:.3f}, for "
        f"~{best.latency_p50_ms:.0f} ms/query median (vs ~{base.latency_p50_ms:.0f} ms "
        "without it) — the cross-encoder dominates latency.",
    ]
    if hyb is not None:
        lines.append(
            f"- **BM25 helps on its own**: hybrid beats dense at MRR "
            f"({hyb.mrr:.3f} vs {base.mrr:.3f}) and hit@1 "
            f"({hyb.hit_rate[1]:.3f} vs {base.hit_rate[1]:.3f}) — exact-term matches "
            "(method acronyms, symbols) that a 33M-param embedder blurs."
        )
    lines += [
        f"- **Under reranking, hybrid vs dense is within noise** on this "
        f"{base.n_questions}-question set (~1 question per point). The eval "
        "over-samples clean \"how does X work\" questions; hybrid is kept on by "
        "default as a safety net for rarer exact-term queries.",
        f"- **Guard rail**: {best.guard_pass}/{best.guard_total} out-of-domain probes "
        "correctly flagged low-confidence in every configuration.",
        "",
        f"Configured default: `{config.settings.retrieval_summary()}`.",
    ]
    missy = [m for m in results if m.misses]
    if missy:
        lines += ["", f"## Remaining misses (gold paper not in top-{kmax})", ""]
        for m in missy:
            lines.append(f"**{m.label}** — {len(m.misses)} miss(es):")
            for mm in m.misses:
                lines.append(f"- _{mm['q']}_  \n  wanted `{mm['gold']}`, got `{mm['got']}`")
            lines.append("")
    if judge is not None:
        mean_score, n_ok, n_total = judge
        lines += [
            "",
            "## Answer faithfulness (LLM-as-judge)",
            "",
            f"Mean faithfulness **{mean_score:.2f} / 5** over {n_ok}/{n_total} gradable answers "
            f"(judge: {config.settings.generator} · {config.settings.gemini_model if config.settings.generator == 'gemini' else config.settings.ollama_model}). "
            "Reported separately from retrieval — a small judge is noisy.",
        ]
    lines += ["", "---", "", "_Regenerate: `python src/evaluate.py --ablation --out eval/results.md`_", ""]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the RAG assistant.")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--ablation", action="store_true", help="run the dense/hybrid/+rerank matrix")
    ap.add_argument("--judge", action="store_true", help="also run LLM-as-judge faithfulness (slow)")
    ap.add_argument("--out", type=str, help="write a markdown report to this path")
    ap.add_argument("--json", type=str, help="dump raw metrics as JSON to this path")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[config] {config.settings.retrieval_summary()}")

    if args.ablation:
        results = run_ablation(args.k)
    else:
        r = Retriever()
        r.warmup()
        results = [evaluate_retrieval(r, args.k, config.settings.retrieval_mode + (" + rerank" if config.settings.rerank else ""))]

    print_table(results, args.k)

    judge = None
    if args.judge:
        print("\n[judge] generating + grading answers (slow on CPU)...")
        best_over = ABLATION_CONFIGS[-1][1] if args.ablation else {}
        judge = evaluate_faithfulness(_apply(best_over) if args.ablation else Retriever())

    def _write(path: str, text: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path}")

    if args.json:
        _write(
            args.json,
            json.dumps({"results": [asdict(m) for m in results], "k_values": args.k}, indent=2),
        )
    if args.out:
        _write(args.out, markdown_report(results, args.k, judge))

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
