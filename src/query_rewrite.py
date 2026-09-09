"""
Multi-query expansion.

Retrieval fails when the user's phrasing doesn't match a paper's own words --
e.g. asking for "three reasons deep nets can't beat GBDTs" when the paper frames
them as "inductive biases" and "challenges". The fix: ask the LLM for a few
alternative phrasings, retrieve for each, and fuse the results (RRF, in
retrieval.py). The original query is always kept as one of the variants.

Config: settings.query_rewrite (off by default -- one extra LLM call per query),
settings.query_rewrite_n.

Failure is non-fatal: if the LLM call errors, this returns just [original] and
retrieval proceeds as normal.
"""

from __future__ import annotations

import time
from functools import lru_cache

from config import settings

# Gemini free tier is ~15 req/min. One rewrite call per user question never
# hits that, but a batch run (the eval) would -- so space successive calls out
# when the backend is Gemini. No-op for Ollama and for the first call.
_MIN_INTERVAL_S = 60.0 / 13
_last_call = [0.0]


def _pace() -> None:
    if settings.generator != "gemini":
        return
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()

_PROMPT = """You rewrite a search query for a corpus of machine-learning papers \
on tabular data (gradient-boosted trees vs deep/attention models).

Produce {n} alternative phrasings of the QUESTION that a relevant paper's own \
abstract or section headings would likely use. Vary the terminology (e.g. \
"inductive bias", "rotation invariance", "feature tokenizer", "ordered target \
statistics"). Keep each on one line, no numbering, no extra text.

QUESTION: {q}

REWRITES:"""


@lru_cache(maxsize=256)
def _rewrite_cached(query: str, n: int, backend: str, model: str) -> tuple[str, ...]:
    # backend/model are in the key only so the cache invalidates if the
    # generator config changes within a process.
    from generation import get_generator

    _pace()
    raw = get_generator().generate(_PROMPT.format(n=n, q=query), stream=False)
    lines = [ln.strip(" -\t").strip() for ln in str(raw).splitlines()]
    variants = [ln for ln in lines if len(ln) > 8][:n]
    # de-dupe, keep the original first
    out, seen = [query], {query.lower()}
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return tuple(out)


def expand(query: str) -> list[str]:
    """Return [original, *alternatives]. Never raises; falls back to [original]."""
    if not settings.query_rewrite:
        return [query]
    try:
        model = settings.gemini_model if settings.generator == "gemini" else settings.ollama_model
        return list(_rewrite_cached(query.strip(), settings.query_rewrite_n, settings.generator, model))
    except Exception:
        return [query]
