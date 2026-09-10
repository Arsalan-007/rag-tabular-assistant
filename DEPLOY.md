# Deploying the demo (Streamlit Community Cloud)

One-time setup, ~10 minutes. Free tier: CPU only, ~1 GB RAM, sleeps when idle.

### 1. Prerequisites
- Repo pushed to GitHub **with its Git LFS object** — the prebuilt vector store.
  Verify: `git lfs ls-files` lists `data/chroma.tar.gz`, and on github.com the
  file shows as an LFS pointer (not raw binary).
- A free Gemini API key: <https://aistudio.google.com/apikey>

### 2. Create the app
1. <https://share.streamlit.io> → **Create app** → **Deploy from GitHub**.
2. Repository `Arsalan-007/rag-tabular-assistant`, branch `main`.
3. **Main file path:** `streamlit_app.py` (repo root). `src/app.py` also works.
4. **Advanced settings → Python version: 3.12.** (`requirements.txt` pins a
   CPU torch wheel that exists for cp311/cp312. Not 3.13.)

### 3. Secrets
**Settings → Secrets**, paste (template in `.streamlit/secrets.toml.example`):

```toml
GENERATOR      = "gemini"
GEMINI_API_KEY = "..."
GEMINI_MODEL   = "gemini-flash-lite-latest"

# The default reranker (bge-reranker-base, ~1.1 GB in memory) does NOT fit the
# free tier. Use the small cross-encoder instead -- ~90 MB, still a real
# reranker, only a little weaker.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

If the app still gets OOM-killed, turn reranking off entirely:

```toml
RERANK = "false"
```

That is a defensible fallback, not a broken one: dense-only retrieval still
scores **hit@5 0.950 / MRR 0.878** on the eval set (see `eval/results.md`) —
reranking mainly buys hit@3 and hit@10.

### 4. Deploy
First build takes several minutes: it installs `requirements.txt` (serving deps
only — the ingestion packages are in `requirements-ingest.txt` and are skipped),
pulls the LFS tarball, and downloads the embedding + reranker models. On boot
`store.ensure_store()` extracts the tarball; **no ingestion runs on the server.**

### 5. Verify
- Masthead status pill reads **ready**.
- Sidebar → Backend shows `Gemini · gemini-flash-lite-latest (API)`.
- Ask an example question: the answer streams, cites `[1] [2]`, and the
  **Sources** panel lists papers with `dense` / `bm25` tags and score bars.
- Ask a follow-up ("how does that compare to X?") — the turn footer should show
  `searched for "<the condensed standalone question>"`.

### Updating
Push to `main`; Streamlit Cloud redeploys automatically. If you rebuilt the
store (`make ingest`), commit the regenerated `data/chroma.tar.gz` (LFS) and it
ships on the next deploy.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Build fails downloading torch, or runs out of disk | Python version isn't 3.11/3.12 — the `+cpu` wheel pin has no wheel for it. Set 3.12 in Advanced settings. |
| App restarts on the first question | OOM from the reranker. Set `RERANKER_MODEL` to the MiniLM cross-encoder, or `RERANK = "false"`. |
| Status pill shows `GEMINI_API_KEY is not set` | Secrets not saved, or the key name differs. It must be exactly `GEMINI_API_KEY`. |
| `No vector store … and no tarball` | LFS object didn't come through. Check `data/chroma.tar.gz` on github.com is a pointer file and LFS is enabled for the repo. |
| Answers say "the passages do not contain…" a lot | Expected for questions outside the 41 papers — the low-confidence banner should also appear. Not a bug. |
