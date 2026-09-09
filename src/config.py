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
    gemini_model: str = "gemini-2.0-flash"

    # ── Embeddings ──────────────────────────────────────────────────────
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # ── Retrieval ──────────────────────────────────────────────────────
    retrieval_mode: str = "hybrid"  # "dense" | "hybrid"
    rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    fetch_k: int = 20  # candidates pulled from each retriever, before fusion
    rerank_candidates: int = 12  # of the fused pool, how many the cross-encoder scores
    top_k: int = 5  # passages handed to the generator
    rrf_k: int = 60  # Reciprocal Rank Fusion smoothing constant
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
        stages = ["bge-small"]
        if self.retrieval_mode == "hybrid":
            stages.append("BM25")
            stages.append(f"RRF(k={self.rrf_k})")
        if self.rerank:
            stages.append(f"rerank[{self.reranker_model.split('/')[-1]}]")
        return f"{' → '.join(stages)}  (fetch_k={self.fetch_k}, top_k={self.top_k})"


settings = Settings()
