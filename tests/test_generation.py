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


def test_gemini_generator_needs_key(monkeypatch):
    monkeypatch.setattr(generation.settings, "gemini_api_key", "")
    ok, msg = make_generator("gemini").health()
    assert ok is False and "GEMINI_API_KEY" in msg
