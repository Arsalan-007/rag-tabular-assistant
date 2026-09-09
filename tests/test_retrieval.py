"""Retrieval-engine tests.

The pure functions (tokenisation, RRF) are tested in isolation. The end-to-end
pipeline tests need the committed Chroma store; they are skipped if it's absent
(e.g. a shallow clone without Git LFS).
"""

from __future__ import annotations

import pytest

from config import settings
from retrieval import Retriever, _tokenize
from store import TARBALL, ensure_store, store_is_materialised

# The store ships as a tarball; extract it if a real one isn't unpacked yet.
_STORE_OK = False
if store_is_materialised() or (TARBALL.exists() and TARBALL.stat().st_size > 10_000):
    try:
        ensure_store()
        _STORE_OK = store_is_materialised()
    except Exception:
        _STORE_OK = False

needs_store = pytest.mark.skipif(not _STORE_OK, reason="vector store unavailable (Git LFS tarball missing)")


# ── pure functions ──────────────────────────────────────────────────────
def test_tokenize_lowercases_and_keeps_alnum():
    assert _tokenize("FT-Transformer uses MA30 and L2!") == [
        "ft", "transformer", "uses", "ma30", "and", "l2",
    ]


def test_tokenize_empty():
    assert _tokenize("   ...  ") == []


def test_rrf_rewards_agreement_across_rankings():
    rrf = Retriever._rrf
    # 'b' is rank 1 in both lists; 'a' rank 2 in both; 'c'/'d' appear once, last.
    fused = rrf([["b", "a", "c"], ["b", "a", "d"]], k=60)
    assert fused["b"] > fused["a"] > fused["c"]
    assert fused["c"] == pytest.approx(fused["d"])  # symmetric single appearance
    assert fused["a"] > fused["d"]  # two mid ranks beat one low rank


def test_rrf_score_formula():
    rrf = Retriever._rrf
    fused = rrf([["x", "y"]], k=10)
    assert fused["x"] == pytest.approx(1 / 11)
    assert fused["y"] == pytest.approx(1 / 12)


def test_rrf_k_dampens_rank_gaps():
    rrf = Retriever._rrf
    small_k = rrf([["a", "b"]], k=1)     # rank matters a lot
    large_k = rrf([["a", "b"]], k=1000)  # rank barely matters
    assert (small_k["a"] - small_k["b"]) > (large_k["a"] - large_k["b"])


def test_cap_per_paper_limits_and_preserves_order():
    r = Retriever.__new__(Retriever)
    r._doc_by_id = {
        "s1": {"arxiv_id": "S"}, "s2": {"arxiv_id": "S"}, "s3": {"arxiv_id": "S"},
        "s4": {"arxiv_id": "S"}, "g1": {"arxiv_id": "G"}, "x1": {"arxiv_id": "X"},
    }
    ids = ["s1", "s2", "s3", "s4", "g1", "x1"]
    assert r._cap_per_paper(ids, cap=2) == ["s1", "s2", "g1", "x1"]
    assert r._cap_per_paper(ids, cap=0) == ids  # 0 disables the cap
    assert r._cap_per_paper(ids, cap=99) == ids


# ── end-to-end (needs the store) ───────────────────────────────────────
@needs_store
def test_dense_retrieval_returns_top_k_hits():
    settings.retrieval_mode, settings.rerank = "dense", False
    r = Retriever()
    res = r.retrieve("gradient boosted trees on tabular data", top_k=5)
    assert len(res.hits) == 5
    assert res.mode == "dense" and res.reranked is False
    assert all(h.dense_similarity is not None for h in res.hits)
    # dense hits come back sorted by similarity
    sims = [h.dense_similarity for h in res.hits]
    assert sims == sorted(sims, reverse=True)


@needs_store
def test_hybrid_surfaces_exact_term_match_dense_can_miss():
    """'NODE' (an acronym) is the kind of token BM25 nails and dense blurs."""
    settings.retrieval_mode, settings.rerank = "hybrid", False
    r = Retriever()
    res = r.retrieve("Neural Oblivious Decision Ensembles NODE oblivious trees", top_k=10)
    assert any(h.arxiv_id == "1909.06312" for h in res.hits)  # the NODE paper


@needs_store
def test_rerank_changes_ordering_and_bounds_scores():
    settings.retrieval_mode, settings.rerank = "hybrid", True
    r = Retriever()
    res = r.retrieve("How does TabNet use sequential attention?", top_k=5)
    assert res.reranked is True
    assert len(res.hits) == 5
    assert all(0.0 <= h.score <= 1.0 for h in res.hits)
    assert [h.score for h in res.hits] == sorted((h.score for h in res.hits), reverse=True)


@needs_store
def test_low_confidence_flag_trips_on_off_topic_query():
    settings.retrieval_mode, settings.rerank = "hybrid", False
    r = Retriever()
    on = r.retrieve("Why do trees outperform deep learning on tabular data?")
    off = r.retrieve("best pizza toppings in Naples")
    assert on.best_dense_similarity > off.best_dense_similarity
    assert on.low_confidence is False
