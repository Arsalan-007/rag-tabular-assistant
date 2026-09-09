"""
Streamlit UI for the tabular-DL research assistant.

    streamlit run src/app.py          # local
    (Streamlit Community Cloud points at this same file)

Design: a precision "research instrument", not a chat toy. Dark slate workspace,
one cyan measurement accent, monospace for data. The source panel shows the
retrieval pipeline's own numbers -- rerank score, which retriever surfaced each
passage, per-stage latency -- because making retrieval legible is the point.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# --- Make sibling modules importable no matter the working directory --------
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Bridge Streamlit secrets -> env BEFORE config is imported --------------
# On Streamlit Cloud the API key + provider live in the app's Secrets, not the
# process env that pydantic-settings reads.
for _key in ("GENERATOR", "GEMINI_API_KEY", "GEMINI_MODEL", "RETRIEVAL_MODE", "RERANK"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:
        pass

import rag  # noqa: E402
from config import settings  # noqa: E402
from generation import get_generator  # noqa: E402

st.set_page_config(
    page_title="Tabular-DL Research Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');
      :root {
        --ink:#0d1117; --slate:#161b22; --line:#232b36;
        --signal:#38e2c8; --signal-dk:#16b8a0;
        --text:#c9d3de; --muted:#6b7684;
      }
      .stApp { background: var(--ink); color: var(--text); }
      #MainMenu, footer, header { visibility: hidden; }
      .block-container { padding-top: 2.4rem; max-width: 1180px; }

      .masthead { border-bottom:1px solid var(--line); padding-bottom:1.1rem; margin-bottom:1.4rem; }
      .masthead .eyebrow {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.22em;
        text-transform:uppercase; color:var(--signal); margin:0 0 0.5rem 0;
      }
      .masthead h1 {
        font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.8rem;
        line-height:1.1; margin:0; color:#f0f4f8; letter-spacing:-0.01em;
      }
      .masthead p {
        font-family:'Inter',sans-serif; color:var(--muted); font-size:0.9rem;
        margin:0.55rem 0 0 0; max-width:64ch;
      }

      .lbl {
        font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.18em;
        text-transform:uppercase; color:var(--muted); margin:0 0 0.6rem 0;
      }

      .answer {
        font-family:'Inter',sans-serif; font-size:1.02rem; line-height:1.72;
        color:var(--text); background:var(--slate); border:1px solid var(--line);
        border-left:2px solid var(--signal); border-radius:6px; padding:1.3rem 1.5rem;
      }

      .src {
        background:var(--slate); border:1px solid var(--line); border-radius:6px;
        padding:0.85rem 1rem; margin-bottom:0.7rem;
      }
      .src .idx { font-family:'IBM Plex Mono',monospace; color:var(--signal); font-weight:500; font-size:0.8rem; }
      .src .ttl {
        font-family:'Space Grotesk',sans-serif; font-weight:500; color:#e4eaf0;
        font-size:0.92rem; margin:0.15rem 0 0.35rem 0;
      }
      .src .meta {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--muted);
        margin-bottom:0.55rem;
      }
      .src .meta a { color:var(--muted); text-decoration:underline dotted; }
      .tag {
        display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
        letter-spacing:0.04em; text-transform:uppercase; padding:0.1rem 0.4rem;
        border:1px solid var(--line); border-radius:4px; color:var(--muted); margin-left:0.4rem;
      }

      .pdf-btn {
        display:inline-block; margin-top:0.85rem; padding:0.4rem 0.9rem;
        font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.06em;
        text-transform:uppercase; color:var(--signal);
        border:1px solid var(--line); border-radius:5px; text-decoration:none; transition:all 0.15s ease;
      }
      .pdf-btn:hover { border-color:var(--signal); background:rgba(56,226,200,0.08); }

      .bar-wrap { display:flex; align-items:center; gap:0.6rem; }
      .bar-track { flex:1; height:4px; background:#0a0e13; border-radius:2px; overflow:hidden; }
      .bar-fill { height:100%; background:linear-gradient(90deg,var(--signal-dk),var(--signal)); border-radius:2px; }
      .bar-val {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--signal);
        min-width:3.2ch; text-align:right;
      }

      .stTextInput textarea, .stTextInput input, .stTextArea textarea {
        background:var(--slate)!important; color:var(--text)!important;
        border:1px solid var(--line)!important; border-radius:6px!important;
        font-family:'Inter',sans-serif!important;
      }
      .stTextArea textarea:focus, .stTextInput input:focus { border-color:var(--signal)!important; }
      .stButton button {
        background:var(--signal); color:#06231e; border:none; border-radius:6px;
        font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:0.01em; padding:0.5rem 1.4rem;
      }
      .stButton button:hover { background:var(--signal-dk); color:#06231e; }

      .pill {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; padding:0.2rem 0.6rem;
        border-radius:20px; border:1px solid var(--line);
      }
      .pill.ok  { color:var(--signal); }
      .pill.bad { color:#ff6b6b; border-color:#3a2226; }
      .telemetry {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--muted);
        margin-top:0.7rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _warm():
    """Load retrieval components + (for Ollama) preload the model, once."""
    rag.get_retriever().warmup()
    return rag.warmup()


ok, msg = rag.health_check()
if ok:
    with st.spinner("Warming up retrieval + generator (first launch only)…"):
        warm_ok, warm_msg = _warm()
    if not warm_ok:
        ok, msg = False, warm_msg
    else:
        msg = "ready"

pill = f'<span class="pill {"ok" if ok else "bad"}">● {msg}</span>'

st.markdown(
    f"""
    <div class="masthead">
      <p class="eyebrow">Retrieval-Augmented · Tabular Deep Learning Corpus</p>
      <h1>Why trees win, and whether attention can catch up.</h1>
      <p>Grounded Q&amp;A over <b>41 arXiv papers</b> on gradient-boosted trees vs
      deep/attention models for tabular data. Hybrid retrieval (bge-small + BM25),
      cross-encoder reranking, cited answers. &nbsp; {pill}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

gen = get_generator()
st.markdown(
    f'<p class="telemetry">pipeline &nbsp;{settings.retrieval_summary()}'
    f'&nbsp; · &nbsp; generator &nbsp;{gen.describe()}</p>',
    unsafe_allow_html=True,
)

if "question" not in st.session_state:
    st.session_state.question = ""

EXAMPLES = [
    "Why do gradient-boosted trees outperform deep learning on tabular data?",
    "How does the FT-Transformer turn features into tokens?",
    "How does CatBoost avoid target leakage when encoding categoricals?",
    "How does TabPFN classify a dataset in one forward pass?",
]

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<p class="lbl">Question</p>', unsafe_allow_html=True)
    question = st.text_area(
        "question", value=st.session_state.question, height=110,
        label_visibility="collapsed",
        placeholder="e.g. What inductive biases let tree ensembles handle irregular target functions?",
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        run = st.button("Ask", use_container_width=True, disabled=not ok)
    with c2:
        k = st.slider("passages retrieved", 3, 10, settings.top_k, label_visibility="collapsed")

    with st.expander("Pipeline controls"):
        mode = st.radio(
            "retrieval", ["dense", "hybrid"],
            index=["dense", "hybrid"].index(settings.retrieval_mode),
            horizontal=True,
        )
        rerank = st.checkbox("cross-encoder rerank", value=settings.rerank)
        settings.retrieval_mode, settings.rerank = mode, rerank
        st.caption(
            "Turn rerank off to feel the latency/quality trade-off. "
            "See `eval/results.md` for the full ablation."
        )

    st.markdown('<p class="lbl" style="margin-top:0.8rem">Try one</p>', unsafe_allow_html=True)
    ex_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.question = ex
            st.rerun()

if run and question.strip():
    with left:
        st.markdown('<p class="lbl" style="margin-top:1.4rem">Answer</p>', unsafe_allow_html=True)
        try:
            result = rag.retrieve(question, k)
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()
        hits = result.hits

        if result.low_confidence:
            st.warning(
                f"Best passage similarity {result.best_dense_similarity:.2f} is below "
                f"{settings.min_similarity:.2f} — the corpus may not cover this question. "
                "The answer below may be thin or off."
            )

        slot = st.empty()
        acc = ""
        g0 = time.perf_counter()
        try:
            with st.spinner("Generating grounded answer…"):
                for tok in rag.generate(rag.build_prompt(question, hits), stream=True):
                    acc += tok
                    slot.markdown(acc + "▍")
            slot.markdown(acc if acc.strip() else "_(model returned no text)_")
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()
        gen_ms = (time.perf_counter() - g0) * 1000

        tset = result.timings_ms
        parts = [f"retrieval {tset.get('total', 0):.0f} ms"]
        if "rerank" in tset:
            parts.append(f"(rerank {tset['rerank']:.0f})")
        parts.append(f"· generation {gen_ms / 1000:.1f} s")
        st.markdown(f'<p class="telemetry">{"  ".join(parts)}</p>', unsafe_allow_html=True)

    with right:
        reranked = result.reranked
        st.markdown(
            f'<p class="lbl">Retrieved sources · {"rerank score" if reranked else "score"}</p>',
            unsafe_allow_html=True,
        )
        scores = [h.score for h in hits]
        lo, hi = (min(scores), max(scores)) if scores else (0.0, 1.0)
        for i, h in enumerate(hits, 1):
            # Reranker scores cluster tightly; spread the bar across the shown set
            # so it stays readable. The number is still the raw relevance.
            frac = (h.score - lo) / (hi - lo) if hi > lo else 1.0
            pct = int(10 + 88 * frac)
            src_tag = "+".join(h.sources)
            st.markdown(
                f"""
                <div class="src">
                  <span class="idx">[{i}]</span><span class="tag">{src_tag}</span>
                  <div class="ttl">{h.title}</div>
                  <div class="meta">{h.authors} · {h.year} ·
                    <a href="https://arxiv.org/abs/{h.arxiv_id}" target="_blank">arXiv:{h.arxiv_id}</a>
                  </div>
                  <div class="bar-wrap">
                    <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                    <span class="bar-val">{h.score:.2f}</span>
                  </div>
                  <a class="pdf-btn" href="https://arxiv.org/pdf/{h.arxiv_id}" target="_blank">View PDF ↗</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
elif not ok:
    st.warning(f"Not ready: {msg}")
