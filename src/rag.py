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

# A conversation turn: (user question, assistant answer).
Turn = tuple[str, str]

_CONDENSE_PROMPT = """Given a conversation and a follow-up question, rewrite the \
follow-up as a STANDALONE question that can be understood without the \
conversation. Resolve pronouns and references ("it", "that method", "the second \
one"). Do NOT answer it. If it is already standalone, return it unchanged.

CONVERSATION:
{history}

FOLLOW-UP: {question}

STANDALONE QUESTION:"""


# --- prompt assembly -----------------------------------------------------
def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        source = f"{h.title} ({h.year}, arXiv:{h.arxiv_id})"
        blocks.append(f'[{i}] From "{source}":\n{h.text}')
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT PASSAGES ===\n{_format_context(hits)}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ===\n"
    )


def build_chat_prompt(question: str, history: list[Turn], hits: list[Hit]) -> str:
    """Prompt for a follow-up turn: system + prior turns + fresh passages.

    The passages are re-retrieved for this turn; earlier answers are context for
    continuity only, and every claim must still be grounded in the numbered
    passages below (which are renumbered fresh each turn).
    """
    if not history:
        return build_prompt(question, hits)
    convo = "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-4:])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "This is a follow-up in an ongoing conversation. Use the conversation "
        "for context, but ground every claim in the numbered CONTEXT PASSAGES "
        "below — they were retrieved for this follow-up and are numbered fresh.\n\n"
        f"=== CONVERSATION SO FAR ===\n{convo}\n\n"
        f"=== CONTEXT PASSAGES ===\n{_format_context(hits)}\n\n"
        f"=== FOLLOW-UP ===\n{question}\n\n"
        f"=== ANSWER ===\n"
    )


def condense_question(question: str, history: list[Turn]) -> str:
    """Rewrite a follow-up into a standalone question for retrieval.
    Non-fatal: returns the original question if the LLM call fails."""
    if not history:
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-4:])
    try:
        out = generate(_CONDENSE_PROMPT.format(history=convo, question=question), stream=False)
        return str(out).strip().strip('"') or question
    except Exception:
        return question


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
    """Full non-streaming pipeline for a single question."""
    result = retrieve(question, top_k)
    text = generate(build_prompt(question, result.hits), stream=False)
    return text, result


def answer_chat(
    question: str, history: list[Turn] | None = None, top_k: int | None = None
) -> tuple[str, RetrievalResult, str]:
    """Follow-up-aware pipeline.

    Condenses (question + history) into a standalone query, retrieves fresh
    passages for it, and generates grounded in those + the conversation.
    Returns (answer_text, RetrievalResult, standalone_query).
    """
    history = history or []
    standalone = condense_question(question, history)
    result = retrieve(standalone, top_k)
    text = generate(build_chat_prompt(question, history, result.hits), stream=False)
    return text, result, standalone


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
