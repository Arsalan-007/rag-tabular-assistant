"""
Orchestration facade: retrieval + prompt assembly + generation.

This module is intentionally thin. The moving parts live in:
    retrieval.py   -- dense / hybrid / rerank pipeline  -> RetrievalResult
    generation.py  -- Ollama / Gemini backends behind one interface
    config.py      -- every tunable

Typical use:
    from rag import answer
    text, result = answer("Why do trees beat deep nets on tabular data?")
    # result.hits -> list[Hit], result.timings_ms, result.low_confidence
"""

from __future__ import annotations

from collections.abc import Iterator

from config import settings
from generation import get_generator
from retrieval import Hit, RetrievalResult, get_retriever, retrieve

TOP_K = settings.top_k  # kept for callers that still read rag.TOP_K

SYSTEM_PROMPT = """You are a research assistant answering questions about the \
machine-learning literature on tabular data: why gradient-boosted trees perform \
well, and how attention/transformer architectures compare.

Answer ONLY using the numbered context passages provided. Ground every claim in \
the passages. When you use a passage, cite it inline by its number, like [1] or \
[2]. If the passages do not contain enough information to answer, say so plainly \
instead of guessing. Be precise and concise; prefer the papers' own terminology."""


# --- prompt assembly -----------------------------------------------------
def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        source = f"{h.title} ({h.year}, arXiv:{h.arxiv_id})"
        blocks.append(f'[{i}] From "{source}":\n{h.text}')
    context = "\n\n".join(blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT PASSAGES ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ===\n"
    )


# --- generation (delegates to the configured backend) ------------------
def generate(prompt: str, stream: bool = False, temperature: float | None = None) -> str | Iterator[str]:
    gen = get_generator()
    if temperature is not None and hasattr(gen, "temperature"):
        gen.temperature = temperature
    return gen.generate(prompt, stream=stream)


def warmup() -> tuple[bool, str]:
    return get_generator().warmup()


# --- one-call convenience ----------------------------------------------
def answer(question: str, top_k: int | None = None) -> tuple[str, RetrievalResult]:
    """Full non-streaming pipeline: returns (answer_text, RetrievalResult)."""
    result = retrieve(question, top_k)
    text = generate(build_prompt(question, result.hits), stream=False)
    return text, result


# --- health ----------------------------------------------------------------
def _get_collection():
    """Back-compat shim for callers that reached into rag for the raw collection."""
    return get_retriever()._get_collection()


def health_check() -> tuple[bool, str]:
    """(ok, message) describing vector store + generator readiness."""
    problems = []
    try:
        n = _get_collection().count()
        if n == 0:
            problems.append("vector store is empty — run `make ingest`")
    except Exception as e:
        problems.append(f"cannot open vector store: {e}")

    gen_ok, gen_msg = get_generator().health()
    if not gen_ok:
        problems.append(gen_msg)

    return (False, "; ".join(problems)) if problems else (True, "ready")


# --- CLI (handy for tuning without the UI) ----------------------------
if __name__ == "__main__":
    ok, msg = health_check()
    print(f"[health] {msg}")
    print(f"[retrieval] {settings.retrieval_summary()}")
    print(f"[generator] {get_generator().describe()}")
    if not ok:
        raise SystemExit(1)
    print("\nAsk a question (Ctrl-C to quit).\n")
    try:
        while True:
            q = input("Q> ").strip()
            if not q:
                continue
            result = retrieve(q)
            if result.low_confidence:
                print(f"  (!) best passage similarity {result.best_dense_similarity} "
                      f"< {settings.min_similarity} — the corpus may not cover this)\n")
            for tok in generate(build_prompt(q, result.hits), stream=True):
                print(tok, end="", flush=True)
            print("\n\nSources:")
            for i, h in enumerate(result.hits, 1):
                print(f"  [{i}] {h.title} ({h.year}) arXiv:{h.arxiv_id}  score={h.score:.3f}")
            print(f"\n  timings: {result.timings_ms}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
