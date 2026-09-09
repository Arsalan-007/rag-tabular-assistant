"""Evaluation-harness tests: metric helpers + eval-set data integrity."""

from __future__ import annotations

import pytest

from eval_questions import EVAL_QUESTIONS, OUT_OF_DOMAIN
from evaluate import _paper_rank_of_first_gold, _parse_judge
from seed_papers import SEED_PAPERS


# ── metric helpers ─────────────────────────────────────────────────────
def test_paper_rank_finds_first_gold():
    papers = ["a", "b", "c", "b", "d"]
    assert _paper_rank_of_first_gold(papers, {"c"}) == 3
    assert _paper_rank_of_first_gold(papers, {"b"}) == 2  # first occurrence
    assert _paper_rank_of_first_gold(papers, {"d", "b"}) == 2  # any gold, earliest


def test_paper_rank_returns_none_on_miss():
    assert _paper_rank_of_first_gold(["a", "b"], {"z"}) is None
    assert _paper_rank_of_first_gold([], {"a"}) is None


@pytest.mark.parametrize(
    "raw,expected_score",
    [
        ('{"score": 5, "reason": "ok"}', 5),
        ('noise before {"score": 3, "reason": "meh"} noise after', 3),
        ("```json\n{\"score\": 1, \"reason\": \"bad\"}\n```", 1),
    ],
)
def test_parse_judge_extracts_score(raw, expected_score):
    score, _ = _parse_judge(raw)
    assert score == expected_score


@pytest.mark.parametrize("raw", ["not json at all", '{"score": 9}', '{"reason": "no score"}', ""])
def test_parse_judge_rejects_bad_output(raw):
    score, reason = _parse_judge(raw)
    assert score is None and "unparseable" in reason


# ── eval-set data integrity (cheap CI guard) ─────────────────────────
def test_every_eval_label_is_in_the_corpus():
    corpus = {pid for pid, _ in SEED_PAPERS}
    for item in EVAL_QUESTIONS:
        assert item["papers"], f"question has no gold paper: {item['q']!r}"
        unknown = set(item["papers"]) - corpus
        assert not unknown, f"{item['q']!r} labels papers not in corpus: {unknown}"


def test_eval_questions_are_unique_and_nonempty():
    qs = [it["q"].strip() for it in EVAL_QUESTIONS]
    assert all(qs)
    assert len(qs) == len(set(qs)), "duplicate eval question"
    assert len(EVAL_QUESTIONS) >= 25  # keep the set from silently shrinking


def test_out_of_domain_probes_exist():
    assert len(OUT_OF_DOMAIN) >= 3
