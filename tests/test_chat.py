"""Conversational (follow-up) prompt + condense logic. Generator is faked."""

from __future__ import annotations

import rag
from retrieval import Hit


def _hit(aid="X", text="passage text"):
    return Hit(text=text, arxiv_id=aid, title="T", authors="A", year="2024", chunk_index=0, score=0.9)


def test_condense_returns_question_verbatim_with_no_history():
    assert rag.condense_question("What is TabNet?", []) == "What is TabNet?"


def test_condense_uses_generator_and_strips_quotes(monkeypatch):
    monkeypatch.setattr(rag, "generate", lambda *a, **k: '  "How does TabNet pick features per step?"  ')
    out = rag.condense_question("does it change per layer?", [("What is TabNet?", "A network.")])
    assert out == "How does TabNet pick features per step?"


def test_condense_falls_back_to_original_on_generator_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(rag, "generate", _boom)
    q = "and what about the second one?"
    assert rag.condense_question(q, [("Q", "A")]) == q


def test_chat_prompt_without_history_is_the_plain_prompt():
    hits = [_hit()]
    assert rag.build_chat_prompt("q", [], hits) == rag.build_prompt("q", hits)


def test_chat_prompt_with_history_includes_prior_turns_and_fresh_context():
    hits = [_hit(aid="2106.01342", text="SAINT does row attention")]
    p = rag.build_chat_prompt(
        "how is that different from column attention?",
        [("What is SAINT?", "A tabular transformer with two attention types.")],
        hits,
    )
    assert "CONVERSATION SO FAR" in p
    assert "What is SAINT?" in p
    assert "FOLLOW-UP" in p
    assert "SAINT does row attention" in p  # fresh passage present
    assert "[1]" in p


def test_chat_prompt_truncates_long_history():
    hits = [_hit()]
    history = [(f"q{i}", f"a{i}") for i in range(10)]
    p = rag.build_chat_prompt("next?", history, hits)
    assert "q9" in p and "q8" in p
    assert "q0" not in p  # only the last few turns are kept
