"""Generator-layer tests. No network: the HTTP call is monkeypatched."""

from __future__ import annotations

import io
import json

import pytest

import generation
from generation import OllamaGenerator, make_generator


def test_make_generator_selects_backend():
    assert make_generator("ollama").name == "ollama"
    assert make_generator("gemini").name == "gemini"
    with pytest.raises(ValueError):
        make_generator("gpt5")


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_ollama_non_streaming(monkeypatch):
    gen = OllamaGenerator(url="http://x", model="m")

    def fake_post(payload, timeout=300):
        assert payload["stream"] is False
        return _FakeResp(json.dumps({"response": "hello world"}).encode())

    monkeypatch.setattr(gen, "_post", fake_post)
    assert gen.generate("prompt", stream=False) == "hello world"


def test_ollama_streaming_yields_tokens(monkeypatch):
    gen = OllamaGenerator(url="http://x", model="m")
    lines = [
        json.dumps({"response": "he"}).encode(),
        json.dumps({"response": "llo"}).encode(),
        json.dumps({"response": "", "done": True}).encode(),
    ]

    monkeypatch.setattr(gen, "_post", lambda payload, timeout=300: _FakeResp(b"\n".join(lines)))
    assert "".join(gen.generate("p", stream=True)) == "hello"


def test_ollama_describe_and_health_message(monkeypatch):
    gen = OllamaGenerator(url="http://nope:1", model="m")
    assert "m" in gen.describe()
    ok, msg = gen.health()
    assert ok is False and "not reachable" in msg


def test_quota_message_distinguishes_daily_from_per_minute():
    from generation import quota_message

    day = "429 {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '500'}"
    minute = "429 {'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '15'}"
    assert "daily" in quota_message(day) and "midnight" in quota_message(day)
    assert "per-minute" in quota_message(minute)
    # the daily case must NOT tell the user to just wait a moment
    assert "Wait about a minute" not in quota_message(day)


def test_gemini_falls_back_to_next_model_when_out_of_quota():
    """Free-tier quota is per project PER MODEL, so another model is fresh budget."""
    from generation import GeminiGenerator

    g = GeminiGenerator(api_key="k", model="primary")
    g._chain = ["primary", "backup"]

    class _Models:
        def generate_content(self, model, **kw):
            if model == "primary":
                raise RuntimeError("429 RESOURCE_EXHAUSTED PerDay")
            return type("R", (), {"text": f"served by {model}"})()

    g._client = type("C", (), {"models": _Models()})()
    assert g.generate("q") == "served by backup"
    assert "backup" in g.describe() and "fell back" in g.describe()


def test_gemini_reraises_non_transient_errors_without_burning_the_chain():
    from generation import GeminiGenerator

    g = GeminiGenerator(api_key="k", model="primary")
    g._chain = ["primary", "backup"]
    tried = []

    class _Models:
        def generate_content(self, model, **kw):
            tried.append(model)
            raise RuntimeError("400 INVALID_ARGUMENT bad prompt")

    g._client = type("C", (), {"models": _Models()})()
    with pytest.raises(RuntimeError):
        g.generate("q")
    assert tried == ["primary"]  # did not pointlessly retry other models


def test_gemini_generator_needs_key(monkeypatch):
    monkeypatch.setattr(generation.settings, "gemini_api_key", "")
    ok, msg = make_generator("gemini").health()
    assert ok is False and "GEMINI_API_KEY" in msg
