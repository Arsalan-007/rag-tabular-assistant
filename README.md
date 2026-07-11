# Tabular-DL Research Assistant (RAG)

A local, offline retrieval-augmented Q&A assistant over the research literature on
**why gradient-boosted trees excel on tabular data, and whether attention/transformer
architectures can beat them.** Built to support the research project
*"MLPs with Attention V2: An End-to-End Boosting-Inspired Neural Architecture."*

Everything runs locally on CPU — no API keys, no cloud, no GPU required.

## Stack

| Component | Choice | Why |
|---|---|---|
| Corpus | ~50 arXiv papers (curated seed + keyword expansion) | The actual literature for this question |
| Parsing | PyMuPDF | Handles two-column scientific PDFs; pure-Python |
| Embeddings | BAAI/bge-small-en-v1.5 | Small, fast on CPU, strong retrieval |
| Vector store | ChromaDB | In-process, persists to disk, zero setup |
| Generator | Mistral 7B via Ollama (Q4_K_M) | Runs on CPU/32GB RAM; swappable for Qwen 2.5 3B |

## Pipeline

```
fetch (arXiv) → parse (strip refs) → chunk (900c / 150 overlap) → embed (bge-small, batched) → store (Chroma)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generator model (separate — install Ollama first: https://ollama.com)
ollama pull mistral        # already have this
# optional faster alternative:
# ollama pull qwen2.5:3b-instruct
```

## Usage

```bash
python src/ingest.py              # fetch seeds + expand to ~50, build the store
python src/ingest.py --no-expand  # curated seed list only
python src/ingest.py --reset      # wipe the store and re-ingest from scratch
```

The pipeline is **resumable** — already-downloaded PDFs and already-embedded
papers are skipped, so re-running after an interruption is safe and cheap.

## Editing the corpus

All paper selection lives in `src/seed_papers.py`:
- `SEED_PAPERS` — the curated arXiv IDs, grouped by research thread
- `NON_ARXIV` — known-relevant papers not on arXiv (reported as gaps; add manually)
- `EXPANSION_QUERIES` / `EXPANSION_TARGET` — keyword-search backfill toward N papers

## A note on missing papers

Four foundational papers (LightGBM, CatBoost's journal version, DeepGBM,
Friedman's original GBM) are **not on arXiv**. The ingestion run reports these
loudly at the end rather than silently omitting them — so the corpus you think
you have matches the corpus you actually have. Download those manually into
`data/pdfs/` to include them.

## Tuning knobs (revisit against the eval set)

In `src/ingest.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBED_BATCH`, `MIN_CHUNK_CHARS`.
Chunk size in particular is worth sweeping once the eval question set exists.

## Roadmap

- [x] Stage 1: ingestion pipeline (this repo)
- [ ] Stage 2: retrieval + generation (query → top-k → Mistral → cited answer)
- [ ] Stage 3: eval set (20–30 questions; retrieval hit-rate + answer quality)
- [ ] Stage 4: interface (CLI or small Streamlit app)
