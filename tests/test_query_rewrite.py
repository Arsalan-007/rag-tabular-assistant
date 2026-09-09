"""Query-expansion tests. No network: the generator is faked."""

from __future__ import annotations

import query_rewrite
from config import settings


def _fake_generator(text):
    class _G:
        def generate(self, prompt, stream=False):
            return text

    return _G()


def test_disabled_returns_only_the_original():
    settings.query_rewrite = False
    assert query_rewrite.expand("why do trees win?") == ["why do trees win?"]


def test_enabled_prepends_original_and_dedupes(monkeypatch):
    settings.query_rewrite = True
    settings.query_rewrite_n = 3
    query_rewrite._rewrite_cached.cache_clear()
    monkeypatch.setattr(
        query_rewrite,
        "get_generator",
        lambda: _fake_generator("- rotation invariance of MLPs\nsmoothness bias\nrotation invariance of MLPs\n"),
        raising=False,
    )
    # get_generator is imported inside the function; patch where it's looked up
    monkeypatch.setattr("generation.get_generator", lambda: _fake_generator(
        "rotation invariance of MLPs\nsmoothness bias of neural nets\nrotation invariance of MLPs"
    ))
    out = query_rewrite.expand("why can't deep nets beat GBDTs?")
    assert out[0] == "why can't deep nets beat GBDTs?"
    assert "smoothness bias of neural nets" in out
    assert len(out) == len(set(o.lower() for o in out))  # no dupes


def test_generator_failure_falls_back_to_original(monkeypatch):
    settings.query_rewrite = True
    query_rewrite._rewrite_cached.cache_clear()

    def _boom():
        raise RuntimeError("api down")

    monkeypatch.setattr("generation.get_generator", lambda: type("G", (), {"generate": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))})())
    assert query_rewrite.expand("q") == ["q"]
