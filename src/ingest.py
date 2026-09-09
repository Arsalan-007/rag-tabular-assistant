"""
Ingestion pipeline for the tabular-DL research assistant.

Stages:
    1. FETCH    download every paper in the frozen corpus (by arXiv ID)
    2. PARSE    extract text from each PDF (PyMuPDF), strip reference lists
    3. CHUNK    split into overlapping, section-aware chunks
    4. EMBED    encode chunks with bge-small (batched, CPU-friendly)
    5. STORE    persist chunks + metadata to a local Chroma collection

Run:  python src/ingest.py          # full pipeline, resumable
      python src/ingest.py --reset   # wipe the vector store and re-ingest

The corpus is the frozen, hand-checked list in src/seed_papers.py -- there is
no keyword expansion (an earlier version had it; it polluted the store with
off-topic papers).

Design notes:
    - Every stage is idempotent: PDFs and embeddings already present are skipped,
      so you can re-run after a crash without redoing work or duplicating rows.
    - Papers that can't be fetched are logged LOUDLY at the end, never dropped
      silently -- the corpus you think you have should match the corpus you have.
"""

import argparse
import os
import re
import time
import urllib.request

import arxiv
import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, PDF_DIR, settings
from seed_papers import NON_ARXIV, SEED_PAPERS

# All tunables live in config.py (env / .env overridable). Bound as module
# names here so the pipeline body reads cleanly.
EMBED_MODEL = settings.embed_model
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
EMBED_BATCH = settings.embed_batch
MIN_CHUNK_CHARS = settings.min_chunk_chars


# ==========================================================================
# STAGE 1 — FETCH
# ==========================================================================
def fetch_papers():
    """Download every paper in the frozen corpus (src/seed_papers.py) by arXiv ID.

    No keyword expansion: the corpus is a hand-checked list. Returns
    (fetched_meta: dict[id -> metadata], missing_ids: list).
    """
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)

    fetched = {}
    missing = []
    seed_ids = [pid for pid, _ in SEED_PAPERS]

    print(f"\n[FETCH] Requesting {len(seed_ids)} corpus papers by arXiv ID...")
    try:
        results = list(client.results(arxiv.Search(id_list=seed_ids)))
    except Exception as e:
        print(f"[FETCH] arXiv batch request failed: {e}")
        results = []

    returned_ids = set()
    for r in results:
        pid = r.get_short_id().split("v")[0]  # strip version suffix
        returned_ids.add(pid)
        fetched[pid] = _download_and_meta(r, pid)

    # any corpus ID arXiv didn't return
    for pid in seed_ids:
        if pid not in returned_ids:
            missing.append(pid)

    return fetched, missing


def _download_and_meta(result, pid):
    """Download one PDF (skip if present) and return its metadata dict."""
    pdf_path = PDF_DIR / f"{pid}.pdf"
    meta = {
        "arxiv_id": pid,
        "title": result.title.strip().replace("\n", " "),
        "authors": ", ".join(a.name for a in result.authors[:6]),
        "year": str(result.published.year) if result.published else "n/a",
        "pdf_path": str(pdf_path),
    }
    if pdf_path.exists():
        print(f"[FETCH]   cached  {pid}  {meta['title'][:60]}")
        return meta
    try:
        # arxiv >=2.x moved download off Result; download from pdf_url directly
        # so we don't depend on the library version. Fall back to the old
        # Result.download_pdf if pdf_url isn't available for some reason.
        pdf_url = getattr(result, "pdf_url", None)
        if pdf_url:
            req = urllib.request.Request(
                pdf_url, headers={"User-Agent": "rag-tabular-assistant/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                pdf_path.write_bytes(resp.read())
        else:
            result.download_pdf(dirpath=str(PDF_DIR), filename=f"{pid}.pdf")
        print(f"[FETCH]   ok      {pid}  {meta['title'][:60]}")
        time.sleep(3)  # arXiv asks for <=1 request / 3s; be polite
    except Exception as e:
        print(f"[FETCH]   FAILED  {pid}: {e}")
        meta["pdf_path"] = None
    return meta


# ==========================================================================
# STAGE 2 — PARSE
# ==========================================================================
# Matches a line that is just a references/bibliography header.
_REF_HEADER = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)


def parse_pdf(pdf_path: str) -> str:
    """Extract text from a PDF and cut everything from the reference list on.
    PyMuPDF's default reading order handles two-column layouts reasonably."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    full = "\n".join(pages)

    # Truncate at the last 'References'/'Bibliography' header if present.
    lines = full.split("\n")
    cut = None
    for i, line in enumerate(lines):
        if _REF_HEADER.match(line):
            cut = i  # keep scanning; use the LAST one (handles per-section refs)
    if cut is not None:
        lines = lines[:cut]
    text = "\n".join(lines)

    # Collapse hyphenated line breaks and excess whitespace.
    text = re.sub(r"-\n(\w)", r"\1", text)      # de-hyphenate wrapped words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ==========================================================================
# STAGE 3 — CHUNK
# ==========================================================================
def chunk_text(text: str):
    """Paragraph-aware sliding window. Accumulate paragraphs up to CHUNK_SIZE,
    then start a new chunk carrying CHUNK_OVERLAP chars of tail context."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for para in paras:
        if len(buf) + len(para) + 1 <= CHUNK_SIZE:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            # start next buffer with overlap tail + this paragraph
            tail = buf[-CHUNK_OVERLAP:] if buf else ""
            buf = f"{tail}\n{para}".strip()
            # a single giant paragraph may still exceed CHUNK_SIZE -> hard-split
            while len(buf) > CHUNK_SIZE:
                chunks.append(buf[:CHUNK_SIZE])
                buf = buf[CHUNK_SIZE - CHUNK_OVERLAP:]
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


# ==========================================================================
# STAGES 4 & 5 — EMBED + STORE
# ==========================================================================
def build_store(fetched_meta, reset: bool):
    """Chunk every parsed paper, embed in batches, and upsert into Chroma."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[STORE] existing collection wiped (--reset)")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    print(f"\n[EMBED] loading embedding model {EMBED_MODEL} (CPU)...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    already = set(collection.get()["ids"]) if collection.count() else set()
    if already:
        print(f"[STORE] {len(already)} chunks already stored -- skipping those")

    total_new = 0
    for meta in fetched_meta.values():
        if not meta.get("pdf_path") or not os.path.exists(meta["pdf_path"]):
            continue
        pid = meta["arxiv_id"]

        # skip papers whose first chunk id is already present
        if f"{pid}::0" in already:
            continue

        try:
            text = parse_pdf(meta["pdf_path"])
        except Exception as e:
            print(f"[PARSE]   FAILED {pid}: {e}")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"[CHUNK]   {pid}: no usable chunks (parse likely poor)")
            continue

        ids = [f"{pid}::{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "arxiv_id": pid,
                "title": meta["title"],
                "authors": meta["authors"],
                "year": meta["year"],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # BGE retrieval works best when passages are embedded as-is (no prefix);
        # the query side gets the "Represent this sentence..." instruction.
        embeddings = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start:start + EMBED_BATCH]
            vecs = model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False
            )
            embeddings.extend(vecs.tolist())

        collection.add(
            ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
        )
        total_new += len(chunks)
        print(f"[STORE]   {pid}: +{len(chunks)} chunks "
              f"({meta['title'][:50]})")

    print(f"\n[STORE] done. {total_new} new chunks; "
          f"collection now holds {collection.count()} total.")
    return collection.count()


# ==========================================================================
# ORCHESTRATION
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Ingest the frozen tabular-DL corpus into a RAG store.")
    ap.add_argument("--reset", action="store_true",
                    help="wipe the vector store before ingesting")
    ap.add_argument("--no-pack", action="store_true",
                    help="skip repacking data/chroma.tar.gz afterwards")
    args = ap.parse_args()

    t0 = time.time()
    fetched, missing = fetch_papers()
    count = build_store(fetched, reset=args.reset)

    if not args.no_pack:
        from store import pack_store
        pack_store()

    # --- final report: what made it in, what didn't -----------------------
    print("\n" + "=" * 68)
    print("INGESTION SUMMARY")
    print("=" * 68)
    print(f"  Papers fetched:        {len(fetched)}")
    print(f"  Chunks in store:       {count}")
    print(f"  Elapsed:               {time.time() - t0:.0f}s")

    if missing:
        print(f"\n  ⚠  {len(missing)} seed paper(s) NOT returned by arXiv "
              f"(check the IDs):")
        for pid in missing:
            label = next((lbl for pid_, lbl in SEED_PAPERS if pid_ == pid), "?")
            print(f"       - {pid}  ({label})")

    if NON_ARXIV:
        print(f"\n  ⚠  {len(NON_ARXIV)} known-relevant paper(s) are NOT on arXiv "
              f"and must be added manually:")
        for name in NON_ARXIV:
            print(f"       - {name}")

    print("\nDone. Next: run the retrieval/query step against the store.")


if __name__ == "__main__":
    main()
