"""
Streamlit UI for the tabular-DL research assistant.

    streamlit run src/app.py          # local
    (Streamlit Community Cloud points at this same file / streamlit_app.py)

A multi-turn chat over the corpus. Each answer is grounded in freshly retrieved
passages and carries its own collapsible source panel, so you can read an
answer, open the cited papers, then ask a follow-up in the same thread.

Theming note: the palette lives in .streamlit/config.toml (Streamlit's own
theme), which is what paints the splash screen, header, chat input and all
widget text. The CSS below is a *thin* layer for the bits Streamlit has no
concept of -- the masthead, source cards, telemetry line. Overriding Streamlit's
internals from CSS is how you end up with unreadable text.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# --- Bridge Streamlit Cloud secrets -> env BEFORE config is imported -------
_SECRET_KEYS = ("GENERATOR", "GEMINI_API_KEY", "GEMINI_MODEL", "RETRIEVAL_MODE", "RERANK")
_secret_files = (Path.home() / ".streamlit/secrets.toml", _HERE.parent / ".streamlit/secrets.toml")
if any(p.exists() for p in _secret_files):
    with contextlib.suppress(Exception):
        for _key in _SECRET_KEYS:
            if _key in st.secrets and _key not in os.environ:
                os.environ[_key] = str(st.secrets[_key])

import rag  # noqa: E402
from config import settings  # noqa: E402
from generation import get_generator  # noqa: E402

st.set_page_config(
    page_title="Tabular-DL Research Assistant",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Thin styling layer. Colours come from .streamlit/config.toml; these are the
# few tokens the custom HTML blocks need, mirrored from it.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

      :root {
        --ink:#0F172A; --panel:#1B2336; --panel-2:#232D42;
        --edge:#334155; --edge-hi:#475569;
        --fg:#E9EEF5; --fg-dim:#94A3B8;
        --teal:#2DD4BF; --teal-soft:rgba(45,212,191,.10);
        --ok:#4ADE80; --warn:#FBBF24;
        --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,monospace;
      }

      html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
      .block-container { padding-top:2rem; padding-bottom:6rem; }
      *:focus-visible { outline:2px solid var(--teal); outline-offset:2px; border-radius:3px; }
      @media (prefers-reduced-motion: reduce) {
        *,*::before,*::after { animation-duration:.01ms!important; transition-duration:.01ms!important; }
      }

      /* ── Masthead ─────────────────────────────────────────────── */
      .masthead { border-bottom:1px solid var(--edge); padding-bottom:1rem; margin-bottom:1.2rem; }
      .masthead .eyebrow {
        font-family:var(--mono); font-size:.66rem; letter-spacing:.14em; text-transform:uppercase;
        color:var(--teal); margin:0 0 .45rem; white-space:nowrap;
      }
      .masthead h1 {
        font-size:1.65rem; font-weight:700; line-height:1.15; margin:0;
        color:#F8FAFC; letter-spacing:-.02em;
      }
      .masthead p {
        color:var(--fg-dim); font-size:.9rem; line-height:1.6; margin:.55rem 0 0; max-width:64ch;
      }
      .pill {
        display:inline-block; font-family:var(--mono); font-size:.68rem; padding:.15rem .55rem;
        border-radius:999px; border:1px solid var(--edge-hi); color:var(--fg-dim);
      }
      .pill.ok  { color:var(--ok);   border-color:rgba(74,222,128,.4); }
      .pill.bad { color:#F87171;     border-color:rgba(248,113,113,.4); }
      .pill .dot { display:inline-block; width:6px; height:6px; border-radius:50%;
                   background:currentColor; margin-right:.4rem; }

      /* ── Chat messages ───────────────────────────────────────── */
      /* Streamlit's default avatars ship loud red/orange fills. Re-tone them
         to the palette rather than hiding them -- they anchor the turn. */
      [data-testid="stChatMessageAvatarUser"],
      [data-testid="stChatMessageAvatarAssistant"] {
        background:var(--panel-2) !important;
        border:1px solid var(--edge) !important;
        color:var(--fg-dim) !important;
        box-shadow:none !important;
      }
      [data-testid="stChatMessageAvatarUser"] {
        border-color:rgba(45,212,191,.45) !important; color:var(--teal) !important;
      }
      [data-testid="stChatMessageAvatarUser"] [data-testid="stIconMaterial"],
      [data-testid="stChatMessageAvatarAssistant"] [data-testid="stIconMaterial"] {
        color:inherit !important;
      }
      [data-testid="stChatMessageContent"] { padding-top:.1rem; }

      /* ── Per-turn footer ─────────────────────────────────────── */
      .turn-meta {
        font-family:var(--mono); font-size:.68rem; font-variant-numeric:tabular-nums;
        color:var(--fg-dim); opacity:.85; margin-top:.6rem; padding-top:.5rem;
        border-top:1px solid var(--edge);
      }
      .turn-meta .rw { color:var(--teal); }

      /* ── Source cards ────────────────────────────────────────── */
      .src-head {
        font-family:var(--mono); font-size:.66rem; letter-spacing:.14em; text-transform:uppercase;
        color:var(--fg-dim); margin:0 0 .6rem;
      }
      .src {
        background:var(--panel-2); border:1px solid var(--edge); border-radius:8px;
        padding:.75rem .9rem; margin-bottom:.6rem; transition:border-color .16s ease;
      }
      .src:hover { border-color:var(--edge-hi); }
      .src .idx { font-family:var(--mono); color:var(--teal); font-weight:500; font-size:.78rem; }
      .src .tag {
        font-family:var(--mono); font-size:.6rem; letter-spacing:.05em; text-transform:uppercase;
        padding:.1rem .4rem; border:1px solid var(--edge-hi); border-radius:4px;
        color:var(--fg-dim); margin-left:.45rem;
      }
      .src .ttl { font-weight:600; color:#F1F5F9; font-size:.88rem; margin:.3rem 0 .25rem; line-height:1.35; }
      .src .meta { font-family:var(--mono); font-size:.68rem; color:var(--fg-dim); margin-bottom:.55rem; }
      .src .meta a { color:var(--fg-dim); text-decoration:underline; text-underline-offset:2px; }
      .bar-wrap { display:flex; align-items:center; gap:.6rem; }
      .bar-track { flex:1; height:4px; background:#0B1220; border-radius:3px; overflow:hidden; }
      .bar-fill { height:100%; background:linear-gradient(90deg,#14B8A6,var(--teal)); border-radius:3px; }
      .bar-val {
        font-family:var(--mono); font-size:.68rem; font-variant-numeric:tabular-nums;
        color:var(--teal); min-width:3ch; text-align:right;
      }
      .pdf-btn {
        display:inline-block; margin-top:.65rem; padding:.34rem .75rem; font-family:var(--mono);
        font-size:.66rem; letter-spacing:.05em; text-transform:uppercase; color:var(--teal);
        border:1px solid var(--edge-hi); border-radius:6px; text-decoration:none;
        transition:background .16s ease, border-color .16s ease;
      }
      .pdf-btn:hover { border-color:var(--teal); background:var(--teal-soft); text-decoration:none; }

      /* "generating…" placeholder, so the answer bubble is never blank */
      .gen-wait { color:var(--fg-dim); font-style:italic; margin:0; animation:pulse 1.5s ease-in-out infinite; }
      @keyframes pulse { 0%,100%{opacity:.45} 50%{opacity:.95} }

      /* skeleton while retrieving */
      .sk { height:64px; border-radius:8px; margin-bottom:.6rem;
            background:linear-gradient(90deg,var(--panel) 25%,var(--panel-2) 37%,var(--panel) 63%);
            background-size:400% 100%; animation:sk 1.3s ease infinite; }
      @keyframes sk { 0%{background-position:100% 0} 100%{background-position:0 0} }

      /* ── Empty state ─────────────────────────────────────────── */
      .empty {
        border:1px solid var(--edge); border-left:2px solid var(--teal); border-radius:8px;
        background:var(--panel); padding:1.1rem 1.3rem; margin-top:.4rem;
      }
      .empty-h {
        font-family:var(--mono); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;
        color:var(--teal); margin:0 0 .7rem;
      }
      .empty ul { margin:0; padding-left:1.1rem; color:var(--fg); font-size:.9rem; line-height:1.9; }
      .empty ul i { color:var(--fg-dim); }
      .empty-f { color:var(--fg-dim); font-size:.83rem; line-height:1.6; margin:.9rem 0 0; }

      /* sidebar section labels */
      .side-h {
        font-family:var(--mono); font-size:.64rem; letter-spacing:.16em; text-transform:uppercase;
        color:var(--fg-dim); margin:1.35rem 0 .5rem;
      }
      .side-h:first-of-type { margin-top:.1rem; }
      /* sidebar captions (pipeline string, backend) read as data -> monospace */
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-family:var(--mono); font-size:.68rem; line-height:1.55; color:var(--fg-dim);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _warm():
    """Load the local models once per server process."""
    rag.get_retriever().warmup()
    return rag.warmup()


@st.cache_data(ttl=120, show_spinner=False)
def _health():
    """Cached deliberately: Streamlit reruns this whole script on EVERY widget
    interaction, so an uncached health check would hit the generator API on
    each slider drag and checkbox toggle -- enough to exhaust a free-tier
    rate limit just by using the controls. TTL so it still recovers on its own."""
    return rag.health_check()


ok, msg = _health()
if ok:
    with st.spinner("Warming up retrieval + generator (first launch only)…"):
        warm_ok, warm_msg = _warm()
    ok, msg = (True, "ready") if warm_ok else (False, warm_msg)

# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "What inductive biases make tree ensembles well-suited to irregular, non-smooth target functions?",
    "How does the FT-Transformer turn numerical and categorical features into tokens?",
    "How does CatBoost reduce target leakage when encoding categorical features?",
    "How does TabPFN classify a small tabular dataset in a single forward pass?",
]

with st.sidebar:
    st.markdown('<p class="side-h">Conversation</p>', unsafe_allow_html=True)
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<p class="side-h">Try one</p>', unsafe_allow_html=True)
    for i, ex in enumerate(EXAMPLES):
        short = ex if len(ex) <= 58 else ex[:55].rstrip() + "…"
        if st.button(short, key=f"ex{i}", use_container_width=True, help=ex):
            st.session_state._pending = ex
            st.rerun()

    st.markdown('<p class="side-h">Retrieval pipeline</p>', unsafe_allow_html=True)
    mode = st.radio(
        "retrieval mode", ["dense", "hybrid"],
        index=["dense", "hybrid"].index(settings.retrieval_mode), horizontal=True,
    )
    rerank_on = st.checkbox("Cross-encoder rerank", value=settings.rerank)
    rewrite_on = st.checkbox(
        "LLM query rewrite", value=settings.query_rewrite,
        help="Expands your question into several phrasings and fuses the results. "
             "Best retrieval scores, but adds an LLM call per turn.",
    )
    k = st.slider("Passages retrieved", 3, 10, settings.top_k)
    settings.retrieval_mode, settings.rerank, settings.query_rewrite = mode, rerank_on, rewrite_on
    st.caption(settings.retrieval_summary())

    st.markdown('<p class="side-h">Backend</p>', unsafe_allow_html=True)
    st.caption(f"{get_generator().describe()} · 41-paper corpus")
    st.caption(
        "[Evaluation results](https://github.com/Arsalan-007/rag-tabular-assistant/blob/main/eval/results.md)"
        " · [Source](https://github.com/Arsalan-007/rag-tabular-assistant)"
    )

# ─────────────────────────────────────────────────────────────────────────
# Masthead
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="masthead">
      <p class="eyebrow">Retrieval-Augmented Generation</p>
      <h1>Why trees win, and whether attention can catch up.</h1>
      <p>Grounded, multi-turn Q&amp;A over <b>41 arXiv papers</b> on gradient-boosted trees vs
      deep&nbsp;/&nbsp;attention models for tabular data. Every answer cites the passages it used.
      &nbsp;<span class="pill {"ok" if ok else "bad"}"><span class="dot"></span>{msg}</span></p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────
def source_cards_html(hits, reranked: bool) -> str:
    scores = [h.score for h in hits] or [0.0]
    lo, hi = min(scores), max(scores)
    cards = []
    for i, h in enumerate(hits, 1):
        frac = (h.score - lo) / (hi - lo) if hi > lo else 1.0
        pct = int(14 + 82 * frac)
        cards.append(
            f"""<div class="src">
              <span class="idx">[{i}]</span><span class="tag">{"+".join(h.sources)}</span>
              <div class="ttl">{h.title}</div>
              <div class="meta">{h.authors} · {h.year} ·
                <a href="https://arxiv.org/abs/{h.arxiv_id}" target="_blank">arXiv:{h.arxiv_id}</a></div>
              <div class="bar-wrap">
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                <span class="bar-val">{h.score:.2f}</span></div>
              <a class="pdf-btn" href="https://arxiv.org/pdf/{h.arxiv_id}" target="_blank">Read PDF ↗</a>
            </div>"""
        )
    label = "ranked by cross-encoder" if reranked else "ranked by retrieval score"
    return f'<p class="src-head">{label}</p>' + "".join(cards)


def render_turn_meta(m: dict) -> None:
    bits = []
    if m.get("standalone") and m["standalone"] != m.get("question"):
        bits.append(f'<span class="rw">searched for</span> “{m["standalone"]}”')
    t = m.get("timings", {})
    if t:
        seg = f'retrieval {t.get("total", 0):.0f}ms'
        if "rerank" in t:
            seg += f' (rerank {t["rerank"]:.0f})'
        bits.append(seg)
    if m.get("gen_ms"):
        bits.append(f'generation {m["gen_ms"] / 1000:.1f}s')
    if bits:
        st.markdown(f'<div class="turn-meta">{" &nbsp;·&nbsp; ".join(bits)}</div>',
                    unsafe_allow_html=True)


def render_sources(m: dict) -> None:
    if m.get("sources"):
        with st.expander(f"Sources ({len(m['sources'])})"):
            st.markdown(source_cards_html(m["sources"], m.get("reranked", False)),
                        unsafe_allow_html=True)


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("low_conf"):
                st.warning(
                    f"Weak corpus match ({m['best_sim']:.2f} < {settings.min_similarity:.2f}). "
                    "This question may fall outside the 41 papers.",
                    icon="⚠️",
                )
            render_sources(m)
            render_turn_meta(m)


# ─────────────────────────────────────────────────────────────────────────
# Conversation
# ─────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty">
          <p class="empty-h">Ask about the corpus</p>
          <ul>
            <li>Mechanisms — <i>“How does CatBoost reduce target leakage?”</i></li>
            <li>Comparisons — <i>“How does SAINT's row attention differ from column attention?”</i></li>
            <li>The core debate — <i>“What inductive biases favour tree ensembles?”</i></li>
          </ul>
          <p class="empty-f">Answers cite the passages they used. Follow-ups keep the thread's
          context, so you can read an answer, open the papers, then dig deeper.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for m in st.session_state.messages:
    render_message(m)

if not ok:
    st.error(f"Not ready — {msg}")

prompt = st.chat_input("Ask a question, or a follow-up…", disabled=not ok)
if not prompt:
    prompt = st.session_state.pop("_pending", None)

if prompt:
    history = [
        (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
        for i in range(0, len(st.session_state.messages) - 1, 2)
        if st.session_state.messages[i]["role"] == "user"
    ]
    st.session_state.messages.append({"role": "user", "content": prompt, "question": prompt})
    render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        standalone = rag.condense_question(prompt, history) if history else prompt

        ph = st.empty()
        ph.markdown('<div class="sk"></div><div class="sk"></div>', unsafe_allow_html=True)
        try:
            result = rag.retrieve(standalone, k)
        except Exception as e:
            ph.error(f"Retrieval failed: {e}")
            st.stop()
        ph.empty()

        if result.low_confidence:
            st.warning(
                f"Weak corpus match ({result.best_dense_similarity:.2f} < "
                f"{settings.min_similarity:.2f}). This question may fall outside the 41 papers.",
                icon="⚠️",
            )

        slot, acc = st.empty(), ""
        # Never leave this empty: if the first token is slow (cold model, or a
        # rate-limited call being retried) an empty bubble looks like a hang.
        slot.markdown('<p class="gen-wait">Generating answer…</p>', unsafe_allow_html=True)
        g0 = time.perf_counter()
        try:
            for tok in rag.generate(rag.build_chat_prompt(prompt, history, result.hits), stream=True):
                acc += tok
                slot.markdown(acc + " ▌")
            slot.markdown(acc if acc.strip() else "_(model returned no text)_")
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                slot.warning(
                    "The generator's free-tier quota is exhausted (limits are per minute "
                    "and per day). The passages below were retrieved successfully — wait "
                    "a moment and ask again.",
                    icon="⚠️",
                )
                render_sources({"sources": result.hits, "reranked": result.reranked})
            else:
                slot.error(f"Generation failed: {err[:300]}")
            st.stop()

        turn = {
            "role": "assistant", "content": acc, "question": prompt, "standalone": standalone,
            "sources": result.hits, "reranked": result.reranked,
            "low_conf": result.low_confidence, "best_sim": result.best_dense_similarity,
            "timings": result.timings_ms, "gen_ms": (time.perf_counter() - g0) * 1000,
        }
        render_sources(turn)
        render_turn_meta(turn)

    st.session_state.messages.append(turn)
