"""
Retrieval engine for the tabular-DL research assistant.

A staged pipeline, each stage toggled from `config.settings`:

    dense      bge-small query embedding  ->  Chroma cosine search   (always)
    sparse     BM25 (Okapi) over the same chunk texts                (hybrid mode)
    fuse       Reciprocal Rank Fusion of the dense + sparse rankings  (hybrid mode)
    rerank     bge-reranker cross-encoder scores the fused candidates (if rerank=on)

Why this shape:

  * Dense retrieval alone misses exact-term matches (method names, symbols like
    "RRF", "CVSS", "MA30"). BM25 catches those. They fail on different queries,
    so fusing their rankings is strictly safer than either alone.
  * RRF fuses *ranks*, not scores, so it needs no score calibration between two
    very differently-scaled retrievers -- that is the whole reason to prefer it
    over a weighted score sum here.
  * A bi-encoder (bge-small) must embed query and passage independently; a
    cross-encoder reads them together and is far more accurate but far slower.
    So we use the cheap retriever to get ~20 candidates and spend the expensive
    cross-encoder only on those. This "retrieve wide, rerank narrow" split is
    the single biggest quality lever in the pipeline (see eval/results.md).

`Retriever.retrieve(query)` returns a `RetrievalResult` with per-stage timings
and a low-confidence flag (best dense cosine below `settings.min_similarity`).
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import BGE_QUERY_PREFIX, CHROMA_DIR, COLLECTION_NAME, settings

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Deliberately dumb: BM25 wants surface forms,
    and scientific text has meaningful alphanumerics (bge, ma30, l2, k=5)."""
    return _WORD.findall(text.lower())


@dataclass
class Hit:
    text: str
    arxiv_id: str
    title: str
    authors: str
    year: str
    chunk_index: int
    score: float  # final-stage relevance in [0, 1]; also the sort key
    dense_similarity: float | None = None  # cosine vs query embedding, if dense-retrieved
    stage_scores: dict = field(default_factory=dict)  # {"dense","bm25_rank","rrf","rerank"}


@dataclass
class RetrievalResult:
    hits: list[Hit]
    mode: str  # "dense" | "hybrid"
    reranked: bool
    best_dense_similarity: float
    low_confidence: bool
    timings_ms: dict


class Retriever:
    """Lazy-loaded, reusable. One instance is enough for a process."""

    def __init__(self) -> None:
        self._embedder: SentenceTransformer | None = None
        self._collection = None
        self._reranker: CrossEncoder | None = None
        # BM25 state, built once from the whole collection
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._doc_by_id: dict[str, dict] = {}

    # --- lazy components -------------------------------------------------
    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.embed_model, device="cpu")
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_collection(COLLECTION_NAME)
        return self._collection

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.reranker_model, device="cpu")
        return self._reranker

    def warmup(self) -> None:
        """Load every component the current config will touch, so the first
        real `retrieve()` call reports steady-state timings (not model load).

        Includes one throwaway vector query: Chroma loads its on-disk HNSW
        index lazily on the first `query()`, which otherwise costs ~3 s.
        """
        emb = self._get_embedder()
        col = self._get_collection()
        with contextlib.suppress(Exception):
            col.query(
                query_embeddings=emb.encode(["warmup"], normalize_embeddings=True).tolist(),
                n_results=1,
            )
        if settings.retrieval_mode == "hybrid":
            self._ensure_bm25()
        if settings.rerank:
            self._get_reranker()

    def _ensure_bm25(self) -> None:
        """Pull every chunk once and build an in-memory BM25 index.

        2.6k chunks of scientific text is a few MB -- cheap to hold and to
        rebuild on process start, and it keeps BM25 exactly in sync with
        whatever is in the vector store."""
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        col = self._get_collection()
        got = col.get(include=["documents", "metadatas"])
        ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
        self._bm25_ids = ids
        self._doc_by_id = {
            i: {"text": d, **m} for i, d, m in zip(ids, docs, metas, strict=True)
        }
        self._bm25 = BM25Okapi([_tokenize(d) for d in docs])

    # --- individual retrievers ---------------------------------------------
    def _dense(self, query: str, k: int) -> tuple[list[str], dict[str, float]]:
        """Return (ranked ids, {id -> cosine similarity})."""
        embedder = self._get_embedder()
        self._last_embed_ms = 0.0
        e0 = time.perf_counter()
        q_vec = embedder.encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True
        ).tolist()
        self._last_embed_ms = (time.perf_counter() - e0) * 1000
        res = self._get_collection().query(
            query_embeddings=q_vec,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        sims = {i: 1.0 - dist for i, dist in zip(ids, res["distances"][0], strict=True)}
        for i, doc, meta in zip(ids, res["documents"][0], res["metadatas"][0], strict=True):
            self._doc_by_id.setdefault(i, {"text": doc, **meta})
        return ids, sims

    def _sparse(self, query: str, k: int) -> list[str]:
        """Return the top-k chunk ids by BM25 score."""
        self._ensure_bm25()
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        return [self._bm25_ids[idx] for idx in ranked[:k]]

    # --- fusion + rerank -------------------------------------------------
    @staticmethod
    def _rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
        """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank_d) over lists."""
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
        return fused

    def _rerank(self, query: str, ids: list[str]) -> list[tuple[str, float]]:
        """Cross-encoder scores, squashed to [0, 1] with a logistic for display."""
        import math

        reranker = self._get_reranker()
        pairs = [(query, self._doc_by_id[i]["text"]) for i in ids]
        raw = reranker.predict(pairs, show_progress_bar=False)
        scored = [(i, 1.0 / (1.0 + math.exp(-float(s)))) for i, s in zip(ids, raw, strict=True)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    # --- public API ---------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or settings.top_k
        fetch_k = max(settings.fetch_k, top_k)
        mode = settings.retrieval_mode
        do_rerank = settings.rerank
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        # dense (always)
        d0 = time.perf_counter()
        dense_ids, dense_sims = self._dense(query, fetch_k)
        t["dense_total"] = (time.perf_counter() - d0) * 1000
        t["embed"] = round(getattr(self, "_last_embed_ms", 0.0), 1)
        t["vector_search"] = round(t["dense_total"] - t["embed"], 1)
        best_dense = max(dense_sims.values()) if dense_sims else 0.0

        # sparse + fuse (hybrid only)
        if mode == "hybrid":
            s0 = time.perf_counter()
            sparse_ids = self._sparse(query, fetch_k)
            t["bm25"] = (time.perf_counter() - s0) * 1000
            fused = self._rrf([dense_ids, sparse_ids], settings.rrf_k)
            candidate_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:fetch_k]
            rrf_scores = fused
        else:
            candidate_ids = dense_ids[:fetch_k]
            rrf_scores = {}

        # rerank (optional) or take the fused/dense head
        if do_rerank:
            r0 = time.perf_counter()
            # Only score the head of the fused pool: the target chunk is almost
            # always there, and cross-encoder passes dominate latency.
            to_rerank = candidate_ids[: settings.rerank_candidates]
            reranked = self._rerank(query, to_rerank)
            t["rerank"] = (time.perf_counter() - r0) * 1000
            final = reranked[:top_k]
            final_scores = dict(reranked)
        else:
            if mode == "hybrid":
                # normalise RRF onto [0,1] just for a comparable display score
                mx = max(rrf_scores.values()) if rrf_scores else 1.0
                final = [(i, rrf_scores[i] / mx) for i in candidate_ids[:top_k]]
            else:
                final = [(i, dense_sims.get(i, 0.0)) for i in candidate_ids[:top_k]]
            final_scores = dict(final)

        t["total"] = (time.perf_counter() - t0) * 1000

        hits: list[Hit] = []
        for doc_id, score in final:
            meta = self._doc_by_id[doc_id]
            hits.append(
                Hit(
                    text=meta["text"],
                    arxiv_id=meta.get("arxiv_id", "?"),
                    title=meta.get("title", "?"),
                    authors=meta.get("authors", "?"),
                    year=str(meta.get("year", "?")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    score=round(float(score), 4),
                    dense_similarity=(
                        round(dense_sims[doc_id], 4) if doc_id in dense_sims else None
                    ),
                    stage_scores={
                        "dense": round(dense_sims[doc_id], 4) if doc_id in dense_sims else None,
                        "rrf": round(rrf_scores[doc_id], 5) if doc_id in rrf_scores else None,
                        "rerank": round(final_scores.get(doc_id), 4) if do_rerank else None,
                    },
                )
            )

        return RetrievalResult(
            hits=hits,
            mode=mode,
            reranked=do_rerank,
            best_dense_similarity=round(best_dense, 4),
            low_confidence=best_dense < settings.min_similarity,
            timings_ms={key: round(val, 1) for key, val in t.items()},
        )


# --- module-level default instance + shims --------------------------------
_default: Retriever | None = None


def get_retriever() -> Retriever:
    global _default
    if _default is None:
        _default = Retriever()
    return _default


def retrieve(query: str, top_k: int | None = None) -> RetrievalResult:
    """Convenience wrapper around the shared Retriever."""
    return get_retriever().retrieve(query, top_k)
