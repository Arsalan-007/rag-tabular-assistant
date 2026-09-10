"""Config validation + corpus integrity."""

from __future__ import annotations

import pytest

from config import Settings
from seed_papers import CORPUS_SIZE, SEED_PAPERS


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.generator == "ollama"
    assert s.retrieval_mode == "hybrid"
    assert s.rerank is True
    assert s.fetch_k >= s.top_k
    assert s.rerank_candidates >= s.top_k


def test_generator_validator_rejects_unknown():
    with pytest.raises(ValueError):
        Settings(_env_file=None, generator="palm")


def test_retrieval_mode_validator_rejects_unknown():
    with pytest.raises(ValueError):
        Settings(_env_file=None, retrieval_mode="sparse")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("RERANK", "false")
    monkeypatch.setenv("GENERATOR", "gemini")
    s = Settings(_env_file=None)
    assert s.top_k == 8
    assert s.rerank is False
    assert s.generator == "gemini"


def test_retrieval_summary_reflects_config():
    dense = Settings(_env_file=None, retrieval_mode="dense", rerank=False)
    full = Settings(_env_file=None, retrieval_mode="hybrid", rerank=True)
    assert "BM25" not in dense.retrieval_summary()
    assert "BM25" in full.retrieval_summary() and "rerank" in full.retrieval_summary()


@pytest.mark.parametrize(
    "raw", ['"AIzaTEST123"', "'AIzaTEST123'", "  AIzaTEST123  ", "AIzaTEST123\n"]
)
def test_secrets_are_stripped_of_quotes_and_whitespace(raw):
    """Values pasted into a secrets UI often arrive quoted or newline-suffixed;
    the API then rejects them as 'invalid key', which reads like a bad key."""
    assert Settings(_env_file=None, gemini_api_key=raw).gemini_api_key == "AIzaTEST123"


def test_corpus_is_frozen_and_unique():
    ids = [pid for pid, _ in SEED_PAPERS]
    assert len(ids) == len(set(ids)) == CORPUS_SIZE
    assert CORPUS_SIZE == 41
