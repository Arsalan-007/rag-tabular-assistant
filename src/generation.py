"""
Answer generation, behind one small interface so the rest of the app never
imports an SDK directly.

    gen = get_generator()                 # picks Ollama or Gemini from config
    text = gen.generate(prompt)           # full string
    for tok in gen.generate(prompt, stream=True):  # token stream
        ...

Two backends:
  OllamaGenerator  -- local, offline, no key. Default for development and for
                      the honest "runs on a laptop" story.
  GeminiGenerator  -- Google Gemini API. Used by the hosted demo, where a local
                      Ollama isn't available.

The prompt is a single pre-assembled string (see rag.build_prompt); neither
backend is given a separate system role, so their outputs stay comparable.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Protocol

from config import settings


class Generator(Protocol):
    name: str

    def describe(self) -> str: ...

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]: ...

    def warmup(self) -> tuple[bool, str]: ...

    def health(self) -> tuple[bool, str]: ...


# ─────────────────────────────────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────────────────────────────────
class OllamaGenerator:
    name = "ollama"

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        keep_alive: int | str = -1,  # -1 = keep model resident for the session
    ) -> None:
        self.url = (url or settings.ollama_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.temperature = settings.temperature if temperature is None else temperature
        self.keep_alive = keep_alive

    def describe(self) -> str:
        return f"Ollama · {self.model} (local)"

    def _post(self, payload: dict, timeout: int = 300):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=timeout)

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        if not stream:
            with self._post(payload) as resp:
                return json.loads(resp.read())["response"]

        def _iter() -> Iterator[str]:
            with self._post(payload) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("response"):
                        yield obj["response"]
                    if obj.get("done"):
                        break

        return _iter()

    def warmup(self) -> tuple[bool, str]:
        """Empty prompt => Ollama just loads the model into RAM and returns."""
        try:
            with self._post(
                {"model": self.model, "prompt": "", "stream": False, "keep_alive": self.keep_alive}
            ) as resp:
                json.loads(resp.read())
            return True, "model loaded"
        except urllib.error.HTTPError as e:
            body = ""
            with contextlib.suppress(Exception):
                body = e.read().decode()[:200]
            return False, f"warm-up failed: HTTP {e.code} {body}"
        except Exception as e:
            return False, f"warm-up failed: {e}"

    def health(self) -> tuple[bool, str]:
        try:
            urllib.request.urlopen(f"{self.url}/api/tags", timeout=5)
            return True, "ollama reachable"
        except Exception:
            return False, f"Ollama not reachable on {self.url} — is it running?"


# ─────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────
class GeminiGenerator:
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.temperature = settings.temperature if temperature is None else temperature
        self._client = None

    def describe(self) -> str:
        return f"Gemini · {self.model} (API)"

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(temperature=self.temperature)

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        client = self._get_client()
        if not stream:
            resp = client.models.generate_content(
                model=self.model, contents=prompt, config=self._config()
            )
            return resp.text or ""

        def _iter() -> Iterator[str]:
            for chunk in client.models.generate_content_stream(
                model=self.model, contents=prompt, config=self._config()
            ):
                if chunk.text:
                    yield chunk.text

        return _iter()

    def warmup(self) -> tuple[bool, str]:
        # Stateless API — nothing to preload. Just confirm the key/model work.
        return self.health()

    def health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GEMINI_API_KEY is not set"
        try:
            client = self._get_client()
            client.models.generate_content(
                model=self.model, contents="ping", config=self._config()
            )
            return True, f"gemini reachable ({self.model})"
        except Exception as e:
            return False, f"Gemini API error: {str(e)[:200]}"


# ─────────────────────────────────────────────────────────────────────────
_BACKENDS = {"ollama": OllamaGenerator, "gemini": GeminiGenerator}
_default: Generator | None = None


def make_generator(name: str | None = None) -> Generator:
    name = (name or settings.generator).lower()
    if name not in _BACKENDS:
        raise ValueError(f"unknown generator {name!r}; choose from {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


def get_generator() -> Generator:
    global _default
    if _default is None:
        _default = make_generator()
    return _default
