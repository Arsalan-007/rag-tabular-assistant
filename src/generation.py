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
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Protocol

from config import settings

# Substrings that mark a transient, worth-retrying API failure (rate limit /
# overloaded / brief 5xx). Anything else propagates immediately.
_TRANSIENT = ("429", "500", "502", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")

# Gemini 429s carry the wait as "Please retry in 12.7s" and "retryDelay": "12s".
_RETRY_HINT = re.compile(r"retry(?:Delay)?[\"']?\s*[:=]?\s*(?:in\s+)?[\"']?(\d+(?:\.\d+)?)\s*s", re.I)


def _retry_transient(
    fn,
    attempts: int = 4,
    base_delay: float = 1.5,
    max_delay: float = 8.0,
    max_total_wait: float = 20.0,
):
    """Call fn(); on a transient error retry with backoff.

    The waiting is deliberately bounded. Gemini's 429s carry a `retryDelay` that
    can be 30-60s; obeying it across several attempts means sleeping for minutes,
    which in an interactive app is indistinguishable from a hang. We'd rather
    fail fast and tell the user than stall silently, so total sleep is capped at
    `max_total_wait` regardless of what the server asks for.
    """
    slept = 0.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised below if not transient / out of tries
            msg = str(e)
            if i == attempts - 1 or not any(t in msg for t in _TRANSIENT):
                raise
            hint = _RETRY_HINT.search(msg)
            wait = min((float(hint.group(1)) + 0.5) if hint else base_delay * (2**i), max_delay)
            if slept + wait > max_total_wait:
                raise
            time.sleep(wait)
            slept += wait


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
            resp = _retry_transient(
                lambda: client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config()
                )
            )
            return resp.text or ""

        def _iter() -> Iterator[str]:
            # Retry only establishing the stream (the HTTP call fires on the
            # first next()); once tokens flow a mid-stream failure propagates.
            def _start():
                it = client.models.generate_content_stream(
                    model=self.model, contents=prompt, config=self._config()
                )
                first = next(it, None)
                return it, first

            try:
                it, first = _retry_transient(_start)
            except Exception as e:
                # The streaming endpoint has its own, tighter free-tier quota:
                # it 429s while plain generateContent still succeeds. Rather
                # than fail the turn, fall back to one non-streaming call and
                # deliver the answer in a single chunk.
                if not any(t in str(e) for t in _TRANSIENT):
                    raise
                resp = client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config()
                )
                if resp.text:
                    yield resp.text
                return

            if first is not None and first.text:
                yield first.text
            for chunk in it:
                if chunk.text:
                    yield chunk.text

        return _iter()

    def warmup(self) -> tuple[bool, str]:
        # Stateless HTTP API: nothing to preload, and a "warm-up" generation
        # would just burn free-tier quota. Verify config instead.
        return self.health()

    def health(self) -> tuple[bool, str]:
        """Validate key + model WITHOUT spending generation quota.

        `models.get` is a metadata call on a separate, far larger quota than
        generateContent. The obvious implementation -- generating a "ping" --
        costs one of the free tier's ~15 requests/minute every time it runs,
        which Streamlit does on every rerun (each widget interaction). That
        alone can 429 a live demo.
        """
        if not self.api_key:
            return False, "GEMINI_API_KEY is not set"
        placeholders = {"<your key>", "your-key-here", "...", "changeme"}
        if self.api_key.lower() in placeholders:
            return False, "GEMINI_API_KEY is still the placeholder text — paste your real key"
        try:
            self._get_client().models.get(model=self.model)
            return True, f"gemini reachable ({self.model})"
        except Exception as e:
            msg = str(e)
            # Turn the two common misconfigurations into actionable messages
            # instead of echoing a raw 400/404 at the user.
            if "API key not valid" in msg or "API_KEY_INVALID" in msg:
                return False, (
                    f"Gemini rejected the API key (length {len(self.api_key)}, "
                    f"starts {self.api_key[:6]!r}). Check Secrets for stray quotes, "
                    "whitespace or a truncated paste — or regenerate the key at "
                    "aistudio.google.com/apikey."
                )
            if "is no longer available" in msg or "NOT_FOUND" in msg:
                return False, (
                    f"Gemini model {self.model!r} is unavailable — set GEMINI_MODEL to a "
                    "current alias such as 'gemini-flash-lite-latest'."
                )
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                return False, (
                    "Gemini free-tier quota is exhausted right now (rate limit is per "
                    "minute and per day). Wait a moment and reload, or point "
                    "GEMINI_MODEL at a model with spare quota."
                )
            return False, f"Gemini API error: {msg[:200]}"


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
