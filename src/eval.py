"""
Evaluation harness for the tabular-DL RAG assistant.

Measures two things, separately (they fail independently):

  RETRIEVAL  -- does the system pull back the right paper?
    * Hit-rate@k : fraction of questions whose correct paper appears in top-k
    * MRR        : mean reciprocal rank of the first correct paper
                   (rewards ranking the right source higher, not just present)
    These numbers are rock-solid: ground truth is the labeled paper ID.

  ANSWER FAITHFULNESS (optional, --judge) -- is the generated answer grounded in
    the retrieved passages? Scored 1-5 by the LLM itself acting as judge.
    Treated as a SEPARATE, noisier signal: a small local judge is imperfect, so
    this is reported apart from the retrieval metrics, never mixed in.

Usage:
    python src/eval.py                 # retrieval metrics only (fast, deterministic)
    python src/eval.py --k 3 5 10      # report hit-rate at multiple k values
    python src/eval.py --judge         # also run answer-faithfulness (slow, needs Ollama)
    python src/eval.py --sweep         # sweep TOP_K to find the best retrieval setting

The retrieval eval needs only the vector store (fast). --judge additionally calls
the LLM once per question, so it takes minutes on CPU.
"""

import argparse
import json
import statistics

import rag


# ==========================================================================
# RETRIEVAL METRICS
# ==========================================================================
def evaluate_retrieval(questions, k_values):
    """For each question, retrieve up to max(k) chunks and record at what rank
    the first correct paper appears. Returns per-question records + aggregates."""
    max_k = max(k_values)
    records = []

    # Sanity: which labeled papers actually exist in the corpus?
    corpus_ids = _corpus_paper_ids()

    for item in questions:
        q = item["q"]
        gold = set(item["papers"])
        missing = gold - corpus_ids
        if missing:
            print(f"  ⚠  label(s) not in corpus for: {q[:60]}  -> {missing}")

        hits = rag.retrieve(q, k=max_k)
        # rank (1-based) of the first retrieved chunk whose paper is a gold paper
        first_hit_rank = None
        retrieved_papers = []
        for rank, h in enumerate(hits, 1):
            retrieved_papers.append(h["arxiv_id"])
            if first_hit_rank is None and h["arxiv_id"] in gold:
                first_hit_rank = rank

        records.append({
            "q": q,
            "gold": sorted(gold),
            "first_hit_rank": first_hit_rank,
            "top_papers": retrieved_papers[:max_k],
        })

    # Aggregate: hit-rate@k for each k, plus MRR.
    agg = {}
    n = len(records)
    for k in k_values:
        hits_at_k = sum(
            1 for r in records
            if r["first_hit_rank"] is not None and r["first_hit_rank"] <= k
        )
        agg[f"hit_rate@{k}"] = hits_at_k / n if n else 0.0

    reciprocals = [
        (1.0 / r["first_hit_rank"]) if r["first_hit_rank"] else 0.0
        for r in records
    ]
    agg["mrr"] = statistics.mean(reciprocals) if reciprocals else 0.0
    return records, agg


def _corpus_paper_ids():
    """All distinct paper IDs currently in the vector store."""
    col = rag._get_collection()
    metas = col.get()["metadatas"]
    return {m["arxiv_id"] for m in metas} if metas else set()


# ==========================================================================
# ANSWER FAITHFULNESS (LLM-as-judge) -- optional, separable
# ==========================================================================
JUDGE_PROMPT = """You are grading whether an ANSWER is faithful to the SOURCE \
passages it was supposedly based on. Faithful means every claim in the answer is \
supported by the sources; unfaithful means the answer adds facts not in the sources \
or contradicts them.

Score 1-5:
5 = fully supported by the sources
4 = mostly supported, minor unsupported detail
3 = partially supported
2 = largely unsupported
1 = contradicts or ignores the sources

Reply with ONLY a JSON object: {"score": <1-5>, "reason": "<one short sentence>"}

=== SOURCES ===
{sources}

=== ANSWER ===
{answer}

=== YOUR JSON GRADE ==="""


def evaluate_answers(questions, k):
    """Generate an answer per question, then have the LLM judge its faithfulness
    to the retrieved sources. Noisy on a small local model -- reported separately."""
    results = []
    for item in questions:
        q = item["q"]
        hits = rag.retrieve(q, k=k)
        answer_text = rag.generate(rag.build_prompt(q, hits), stream=False)

        sources = "\n\n".join(
            f"[{i}] {h['text']}" for i, h in enumerate(hits, 1)
        )
        judge_raw = rag.generate(
            JUDGE_PROMPT.replace("{sources}", sources).replace("{answer}", answer_text),
            stream=False,
        )
        score, reason = _parse_judge(judge_raw)
        results.append({"q": q, "score": score, "reason": reason})
        print(f"  [{score if score else '?'}/5] {q[:60]}")

    scored = [r["score"] for r in results if r["score"] is not None]
    mean_score = statistics.mean(scored) if scored else None
    return results, mean_score


def _parse_judge(raw):
    """Pull {"score", "reason"} out of the judge's reply, tolerating stray text."""
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
        score = int(obj.get("score"))
        if 1 <= score <= 5:
            return score, str(obj.get("reason", ""))[:120]
    except Exception:
        pass
    return None, "unparseable judge output"


# ==========================================================================
# REPORTING
# ==========================================================================
def print_retrieval_report(records, agg, k_values):
    print("\n" + "=" * 68)
    print("RETRIEVAL EVALUATION")
    print("=" * 68)
    for k in k_values:
        print(f"  Hit-rate@{k:<3} {agg[f'hit_rate@{k}']:.3f}")
    print(f"  MRR         {agg['mrr']:.3f}")
    print(f"  Questions   {len(records)}")

    # Show the misses -- the most useful part for tuning.
    misses = [r for r in records if r["first_hit_rank"] is None]
    if misses:
        print(f"\n  Misses (correct paper not in top {max(k_values)}):")
        for r in misses:
            print(f"    - {r['q'][:64]}")
            print(f"        wanted {r['gold']}, got {r['top_papers'][:3]}...")
    else:
        print(f"\n  No misses at k={max(k_values)}.")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    from eval_questions import EVAL_QUESTIONS

    ap = argparse.ArgumentParser(description="Evaluate the RAG assistant.")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 10],
                    help="k values for hit-rate (default: 3 5 10)")
    ap.add_argument("--judge", action="store_true",
                    help="also run LLM-as-judge answer faithfulness (slow)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep TOP_K retrieval depth and report hit-rate/MRR for each")
    args = ap.parse_args()

    ok, msg = rag.health_check()
    print(f"[health] {msg}")
    if "vector store" in msg:
        raise SystemExit("Vector store not ready -- run ingest.py first.")

    if args.sweep:
        print("\nSweeping retrieval depth (k)...")
        print(f"  {'k':>4} | {'hit-rate@k':>10} | {'MRR':>6}")
        for k in [1, 3, 5, 8, 10, 15]:
            _, agg = evaluate_retrieval(EVAL_QUESTIONS, [k])
            print(f"  {k:>4} | {agg[f'hit_rate@{k}']:>10.3f} | {agg['mrr']:>6.3f}")
        return

    records, agg = evaluate_retrieval(EVAL_QUESTIONS, args.k)
    print_retrieval_report(records, agg, args.k)

    if args.judge:
        print("\n" + "=" * 68)
        print("ANSWER FAITHFULNESS (LLM-as-judge, separate/noisier signal)")
        print("=" * 68)
        print("  Generating and grading answers (slow on CPU)...")
        _, mean_score = evaluate_answers(EVAL_QUESTIONS, k=rag.TOP_K)
        if mean_score is not None:
            print(f"\n  Mean faithfulness: {mean_score:.2f} / 5  "
                  f"(over {len(EVAL_QUESTIONS)} answers)")
        else:
            print("\n  Could not compute mean (judge outputs unparseable).")

    print("\nDone.")


if __name__ == "__main__":
    main()
