"""
Central configuration for the RAG assistant.

Every tunable lives here with a sensible default. A local `.env` (see
`.env.example`) or real environment variables override any field. Import the
singleton `settings` everywhere rather than re-reading os.environ.

    from config import settings
    settings.top_k          # -> 5
    settings.retrieval_mode # -> "hybrid"
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Fixed paths (not user-tunable) --------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "tabular_papers"

# bge retrieval works best when the *query* carries this instruction and the
# stored passages do not. Kept here so ingestion and retrieval can't disagree.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Generator ────────────────────────────────────────────────────────
    generator: str = "ollama"  # "ollama" | "gemini"
    temperature: float = 0.2

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-lite-latest"

    # ── Embeddings ──────────────────────────────────────────────────────
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # ── Retrieval ──────────────────────────────────────────────────────
    retrieval_mode: str = "hybrid"  # "dense" | "hybrid"
    rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    fetch_k: int = 20  # candidates pulled from each retriever, before fusion
    # How many fused candidates the cross-encoder scores. 12 measured best on the
    # eval set; raising it to 20 regressed hit@1 / MRR (more distractors to rank).
    rerank_candidates: int = 12
    # Cap chunks-per-paper in the fused pool. 0 = off. A cap of 3 was tried to
    # stop verbose surveys flooding the pool; it regressed the aggregate without
    # fixing the target query, so it's off by default (kept as a knob).
    chunks_per_paper: int = 0
    top_k: int = 5  # passages handed to the generator
    rrf_k: int = 60  # Reciprocal Rank Fusion smoothing constant

    # Multi-query expansion: rewrite the question a few ways with an LLM and
    # fuse the retrievals. Helps the *class* of oblique paraphrases but adds an
    # LLM call per query and can inflate the low-confidence guard, so off by
    # default. When on, the guard still uses the ORIGINAL query's similarity.
    query_rewrite: bool = False
    query_rewrite_n: int = 3  # number of alternative phrasings to generate
    # bge-small embeddings sit around ~0.45-0.55 cosine even for unrelated text
    # and ~0.80+ for on-topic; ~0.6 cleanly separates the two. Calibrated
    # against the eval set + off-topic probes (see eval/results.md).
    min_similarity: float = 0.6  # warn if the best passage scores below this

    # ── Ingestion ──────────────────────────────────────────────────────
    chunk_size: int = 900
    chunk_overlap: int = 150
    embed_batch: int = 48
    min_chunk_chars: int = 200

    @field_validator("generator")
    @classmethod
    def _valid_generator(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"ollama", "gemini"}:
            raise ValueError(f"generator must be 'ollama' or 'gemini', got {v!r}")
        return v

    @field_validator("retrieval_mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"dense", "hybrid"}:
            raise ValueError(f"retrieval_mode must be 'dense' or 'hybrid', got {v!r}")
        return v

    def retrieval_summary(self) -> str:
        """One-line description of the active retrieval pipeline, for logs / UI."""
        stages = []
        if self.query_rewrite:
            stages.append(f"rewrite x{self.query_rewrite_n}")
        stages.append("bge-small")
        if self.retrieval_mode == "hybrid":
            stages.append("BM25")
            stages.append(f"RRF(k={self.rrf_k})")
        if self.chunks_per_paper:
            stages.append(f"cap {self.chunks_per_paper}/paper")
        if self.rerank:
            stages.append(f"rerank[{self.reranker_model.split('/')[-1]}]")
        return f"{' → '.join(stages)}  (fetch_k={self.fetch_k}, top_k={self.top_k})"


settings = Settings()
