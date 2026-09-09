# Deploying the demo (Streamlit Community Cloud)

One-time setup. Free tier: ~1 GB RAM, CPU only, sleeps after inactivity.

### 1. Prerequisites
- The repo is pushed to GitHub **with Git LFS objects** (the vector store tarball).
  Verify: `git lfs ls-files` shows `data/chroma.tar.gz`, and the file on GitHub
  shows as an LFS pointer, not 17 MB of binary.
- A free Gemini API key: <https://aistudio.google.com/apikey>

### 2. Create the app
1. Go to <https://share.streamlit.io> → **New app** → pick this repo / branch.
2. **Main file path:** `streamlit_app.py` (repo root) — or `src/app.py`, both work.
3. **Advanced settings → Python version:** 3.11.

### 3. Secrets
In the app's **Settings → Secrets**, paste (see `.streamlit/secrets.toml.example`):

```toml
GENERATOR = "gemini"
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-flash-lite-latest"
```

Optional, if the ~2 s/query reranker is too slow on the free tier:

```toml
RERANK = "false"
```

### 4. Deploy
Click **Deploy**. First build takes a few minutes: it installs
`requirements.txt`, pulls the LFS tarball, and downloads the embedding +
reranker models (~1.3 GB, cached afterwards). `store.ensure_store()` extracts
the tarball on first request — no ingestion runs on the server.

### 5. Verify
- The status pill in the masthead reads **ready**.
- The telemetry line shows `generator  Gemini · gemini-flash-lite-latest (API)`.
- Ask an example question; sources appear with scores and `dense` / `bm25` tags.

### Updating
Push to the deployed branch — Streamlit Cloud redeploys automatically.
If you rebuilt the store (`make ingest`), commit the new `data/chroma.tar.gz`
(LFS) and it ships on the next deploy.
