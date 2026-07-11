"""
Retrieval + generation backend for the tabular-DL research assistant.

Pipeline per question:
    embed query (bge with query prefix) -> retrieve top-k chunks from Chroma
    -> build grounded prompt -> call Mistral via Ollama -> return answer + sources

Importable by the Streamlit UI (app.py) or usable directly. Keeping this
separate from the UI means retrieval can be tested and tuned on its own.
"""

import json
import urllib.request
import urllib.error
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# --- Config ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "tabular_papers"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
# Keep the model resident in RAM instead of unloading after idle. -1 (integer)
# = never unload while Ollama runs, so you pay the load cost once per session,
# not per question. A string like "30m" would keep it for 30 minutes instead.
OLLAMA_KEEP_ALIVE = -1

# bge query-side instruction. Passages were embedded bare at ingestion, so
# only the query carries this prefix -- required for correct bge retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 5  # chunks fed to the model as context

SYSTEM_PROMPT = """You are a research assistant answering questions about the \
machine-learning literature on tabular data: why gradient-boosted trees perform \
well, and how attention/transformer architectures compare.

Answer ONLY using the numbered context passages provided. Ground every claim in \
the passages. When you use a passage, cite it inline by its number, like [1] or \
[2]. If the passages do not contain enough information to answer, say so plainly \
instead of guessing. Be precise and concise; prefer the papers' own terminology."""


# --- Lazy singletons (load once, reuse) ------------------------------------
_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# --- Retrieval -------------------------------------------------------------
def retrieve(question: str, k: int = TOP_K):
    """Embed the question and return the top-k chunks with their metadata."""
    embedder = _get_embedder()
    collection = _get_collection()

    q_vec = embedder.encode(
        [QUERY_PREFIX + question], normalize_embeddings=True
    ).tolist()

    res = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "title": meta.get("title", "?"),
                "authors": meta.get("authors", "?"),
                "year": meta.get("year", "?"),
                "arxiv_id": meta.get("arxiv_id", "?"),
                "score": 1.0 - dist,  # cosine distance -> similarity
            }
        )
    return hits


# --- Prompt assembly -------------------------------------------------------
def build_prompt(question: str, hits: list) -> str:
    context_blocks = []
    for i, h in enumerate(hits, 1):
        source = f"{h['title']} ({h['year']}, arXiv:{h['arxiv_id']})"
        context_blocks.append(f"[{i}] From \"{source}\":\n{h['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT PASSAGES ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ===\n"
    )


# --- Generation (Ollama) ---------------------------------------------------
def generate(prompt: str, stream: bool = False, temperature: float = 0.2):
    """Call Ollama. If stream=True, yields text chunks; else returns full text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )

    if not stream:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["response"]

    def _stream():
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("response"):
                    yield obj["response"]
                if obj.get("done"):
                    break

    return _stream()


# --- One-call convenience --------------------------------------------------
def answer(question: str, k: int = TOP_K):
    """Full non-streaming pipeline: returns (answer_text, hits)."""
    hits = retrieve(question, k)
    prompt = build_prompt(question, hits)
    text = generate(prompt, stream=False)
    return text, hits


# --- Warm-up ---------------------------------------------------------------
def warmup(timeout: int = 300):
    """Preload the model into RAM so the first real question is fast.
    An empty prompt tells Ollama to just load the model and return.
    Returns (ok, message). Safe to call repeatedly."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "",           # empty prompt = load model only, no generation
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            json.loads(resp.read())
        return True, "model loaded"
    except urllib.error.HTTPError as e:
        # Surface Ollama's actual explanation, not just the status code.
        try:
            body = e.read().decode()
        except Exception:
            body = ""
        return False, f"warm-up failed: HTTP {e.code} {body[:200]}"
    except Exception as e:
        return False, f"warm-up failed: {e}"


def health_check():
    """Return (ok: bool, message) describing store + Ollama readiness."""
    problems = []
    try:
        n = _get_collection().count()
        if n == 0:
            problems.append("vector store is empty — run ingest.py first")
    except Exception as e:
        problems.append(f"cannot open vector store: {e}")
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        problems.append("Ollama not reachable on :11434 — is it running?")
    if problems:
        return False, "; ".join(problems)
    return True, "ready"


# --- CLI fallback (handy for tuning without the UI) ------------------------
if __name__ == "__main__":
    ok, msg = health_check()
    print(f"[health] {msg}")
    if not ok:
        raise SystemExit(1)
    print("Ask a question (Ctrl-C to quit).\n")
    try:
        while True:
            q = input("Q> ").strip()
            if not q:
                continue
            hits = retrieve(q)
            print()
            for tok in generate(build_prompt(q, hits), stream=True):
                print(tok, end="", flush=True)
            print("\n\nSources:")
            for i, h in enumerate(hits, 1):
                print(f"  [{i}] {h['title']} ({h['year']}) "
                      f"arXiv:{h['arxiv_id']}  sim={h['score']:.3f}")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")