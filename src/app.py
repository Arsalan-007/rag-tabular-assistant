"""
Streamlit UI for the tabular-DL research assistant.

Run from the project root:
    streamlit run src/app.py

Design: a precision "research instrument" rather than a chat toy. Dark slate
workspace, a single cyan measurement-accent, monospace for data/metadata, and a
source panel that shows live retrieval similarity as signal bars -- making
retrieval quality visible is the point, not decoration.
"""

import streamlit as st

import rag

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tabular-DL Research Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Design tokens, injected as CSS ----------------------------------------
# Palette (named):
#   ink       #0d1117  base workspace
#   slate     #161b22  raised panels
#   line      #232b36  hairline borders
#   signal    #38e2c8  cyan measurement accent (used with restraint)
#   signal-dk #16b8a0  accent pressed/hover
#   text      #c9d3de  primary text
#   muted     #6b7684  captions, metadata
# Type: 'Space Grotesk' display, 'Inter' body, 'IBM Plex Mono' data.
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

      /* Masthead */
      .masthead { border-bottom:1px solid var(--line); padding-bottom:1.1rem; margin-bottom:1.6rem; }
      .masthead .eyebrow {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.22em;
        text-transform:uppercase; color:var(--signal); margin:0 0 0.5rem 0;
      }
      .masthead h1 {
        font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.85rem;
        line-height:1.1; margin:0; color:#f0f4f8; letter-spacing:-0.01em;
      }
      .masthead p {
        font-family:'Inter',sans-serif; color:var(--muted); font-size:0.9rem;
        margin:0.55rem 0 0 0; max-width:60ch;
      }

      /* Section labels */
      .lbl {
        font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.18em;
        text-transform:uppercase; color:var(--muted); margin:0 0 0.6rem 0;
      }

      /* Answer surface */
      .answer {
        font-family:'Inter',sans-serif; font-size:1.02rem; line-height:1.72;
        color:var(--text); background:var(--slate); border:1px solid var(--line);
        border-left:2px solid var(--signal); border-radius:6px; padding:1.3rem 1.5rem;
      }
      .answer :is(sup,.cite){ color:var(--signal); font-weight:600; }

      /* Source cards */
      .src {
        background:var(--slate); border:1px solid var(--line); border-radius:6px;
        padding:0.85rem 1rem; margin-bottom:0.7rem;
      }
      .src .idx {
        font-family:'IBM Plex Mono',monospace; color:var(--signal); font-weight:500;
        font-size:0.8rem;
      }
      .src .ttl {
        font-family:'Space Grotesk',sans-serif; font-weight:500; color:#e4eaf0;
        font-size:0.92rem; margin:0.15rem 0 0.35rem 0;
      }
      .src .meta {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--muted);
        margin-bottom:0.55rem;
      }
      .src .meta a { color:var(--muted); text-decoration:underline dotted; }

      /* View PDF button inside source cards */
      .pdf-btn {
        display:inline-block; margin-top:0.85rem; padding:0.4rem 0.9rem;
        font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.06em;
        text-transform:uppercase; color:var(--signal);
        border:1px solid var(--line); border-radius:5px;
        text-decoration:none; transition:all 0.15s ease;
      }
      .pdf-btn:hover { border-color:var(--signal); background:rgba(56,226,200,0.08); }

      /* Similarity signal bar -- the signature element */
      .bar-wrap { display:flex; align-items:center; gap:0.6rem; }
      .bar-track {
        flex:1; height:4px; background:#0a0e13; border-radius:2px; overflow:hidden;
      }
      .bar-fill { height:100%; background:linear-gradient(90deg,var(--signal-dk),var(--signal)); border-radius:2px; }
      .bar-val {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--signal);
        min-width:3.2ch; text-align:right;
      }

      /* Inputs */
      .stTextInput textarea, .stTextInput input, .stTextArea textarea {
        background:var(--slate)!important; color:var(--text)!important;
        border:1px solid var(--line)!important; border-radius:6px!important;
        font-family:'Inter',sans-serif!important;
      }
      .stTextArea textarea:focus, .stTextInput input:focus { border-color:var(--signal)!important; }
      .stButton button {
        background:var(--signal); color:#06231e; border:none; border-radius:6px;
        font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:0.01em;
        padding:0.5rem 1.4rem;
      }
      .stButton button:hover { background:var(--signal-dk); color:#06231e; }

      /* Status pill */
      .pill {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; padding:0.2rem 0.6rem;
        border-radius:20px; border:1px solid var(--line);
      }
      .pill.ok  { color:var(--signal); }
      .pill.bad { color:#ff6b6b; border-color:#3a2226; }

      /* Example chips */
      .chips { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Warm up the model once per session (cached) ---------------------------
# This preloads Mistral into RAM at startup so the first real question is fast,
# instead of the user waiting ~2 min on their first click. Cached so it runs
# only once, not on every rerun.
@st.cache_resource(show_spinner=False)
def _warm_model():
    return rag.warmup()

# --- Masthead --------------------------------------------------------------
ok, msg = rag.health_check()

# If the store + Ollama are reachable, warm the model (shows a spinner once).
if ok:
    with st.spinner("Loading the model into memory (first launch only)…"):
        warm_ok, warm_msg = _warm_model()
    if warm_ok:
        msg = "ready"
    else:
        ok, msg = False, warm_msg

pill = (f'<span class="pill ok">● {msg}</span>' if ok
        else f'<span class="pill bad">● {msg}</span>')

st.markdown(
    f"""
    <div class="masthead">
      <p class="eyebrow">Retrieval-Augmented · Tabular Deep Learning Corpus</p>
      <h1>Why trees win, and whether attention can catch up.</h1>
      <p>Ask questions against a corpus of ~50 papers on gradient-boosted trees,
      transformer architectures for tabular data, and boosting-inspired networks.
      Every answer is grounded in retrieved passages and shows its sources. &nbsp; {pill}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Session state ---------------------------------------------------------
if "question" not in st.session_state:
    st.session_state.question = ""

EXAMPLES = [
    "Why do gradient-boosted trees outperform deep learning on tabular data?",
    "How does the FT-Transformer tokenize features?",
    "What makes GrowNet a boosting-inspired neural architecture?",
    "How does SAINT's inter-sample attention differ from feature attention?",
]

# --- Query surface ---------------------------------------------------------
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
        k = st.slider("passages retrieved", 3, 10, rag.TOP_K,
                      label_visibility="collapsed")

    st.markdown('<p class="lbl" style="margin-top:0.8rem">Try one</p>',
                unsafe_allow_html=True)
    ex_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.question = ex
            st.rerun()

# --- Run the pipeline ------------------------------------------------------
if run and question.strip():
    with left:
        st.markdown('<p class="lbl" style="margin-top:1.4rem">Answer</p>',
                    unsafe_allow_html=True)
        try:
            result = rag.retrieve(question, k)
            hits = result.hits
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

        if result.low_confidence:
            st.warning(
                f"Best passage similarity {result.best_dense_similarity:.2f} is low — "
                "the corpus may not cover this question."
            )

        prompt = rag.build_prompt(question, hits)

        # Stream the answer. Any Ollama error is surfaced, not swallowed.
        slot = st.empty()
        acc = ""
        try:
            with st.spinner("Thinking…"):
                for tok in rag.generate(prompt, stream=True):
                    acc += tok
                    slot.markdown(acc + "▍")
            slot.markdown(acc if acc.strip() else "_(model returned no text)_")
        except Exception as e:
            st.error(f"Generation failed (is Ollama running with the model?): {e}")
            st.stop()

    with right:
        st.markdown('<p class="lbl">Retrieved sources · similarity</p>',
                    unsafe_allow_html=True)
        for i, h in enumerate(hits, 1):
            pct = max(0, min(100, round(h.score * 100)))
            arxiv_id = h.arxiv_id
            st.markdown(
                f"""
                <div class="src">
                  <span class="idx">[{i}]</span>
                  <div class="ttl">{h.title}</div>
                  <div class="meta">{h.authors} · {h.year} ·
                    <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">arXiv:{arxiv_id}</a>
                  </div>
                  <div class="bar-wrap">
                    <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                    <span class="bar-val">{h.score:.2f}</span>
                  </div>
                  <a class="pdf-btn" href="https://arxiv.org/pdf/{arxiv_id}" target="_blank">View PDF ↗</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
elif not ok:
    st.warning(f"Not ready: {msg}")