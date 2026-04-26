"""
app.py — Ask First · Clary  |  Streamlit UI
LangGraph-powered health reasoning engine with cross-session temporal pattern detection.
"""

import streamlit as st
import json
import os
from datetime import datetime

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title = "Ask First · Clary",
    page_icon  = "🩺",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background: #f7f6f2 !important;
    color: #28251d !important;
}

/* ── Brand ── */
.brand        { font-size:1.9rem; font-weight:800; color:#28251d; letter-spacing:-0.04em; margin-bottom:0px; }
.brand span   { color:#01696f; background: linear-gradient(135deg, #01696f 0%, #039ba4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.brand-sub    { color:#7a7974; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }

/* ── Sticky Footer ── */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f7f6f2;
    color: #7a7974;
    text-align: center;
    padding: 10px 0;
    font-size: 0.78rem;
    font-weight: 500;
    border-top: 1px solid rgba(40,37,29,0.08);
    z-index: 999;
}

/* ── Sidebar card ── */
.sidebar-user {
    background: #fff;
    border: 1px solid rgba(40,37,29,0.09);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-top: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.sidebar-user-name { font-weight:700; font-size:0.95rem; color:#28251d; }
.sidebar-user-meta { color:#7a7974; font-size:0.78rem; margin-top:2px; }

/* ── Pattern card ── */
.pcard {
    background: #fff;
    border: 1px solid rgba(40,37,29,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.18s;
}
.pcard:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.08); }

.pcard-title { font-size:0.97rem; font-weight:700; color:#28251d; margin-bottom:8px; }

.badge {
    display:inline-block; padding:3px 11px; border-radius:99px;
    font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
}
.bvh { background:#d4efcc; color:#1a6608; }
.bh  { background:#cde8e5; color:#015d63; }
.bm  { background:#fef3cd; color:#7a5a00; }
.bl  { background:#fde8e8; color:#a12c2c; }

.chip {
    display:inline-block; background:#cedcd8; color:#01696f;
    border-radius:99px; padding:2px 9px; font-size:0.68rem; font-weight:600; margin:2px;
}
.tag {
    display:inline-block; background:#edeae5; color:#7a7974;
    border-radius:6px; padding:1px 7px; font-size:0.68rem; margin:2px;
}
.label { font-size:0.72rem; font-weight:600; text-transform:uppercase;
         letter-spacing:0.06em; color:#7a7974; margin-top:10px; margin-bottom:3px; }

/* ── Reasoning trace ── */
.trace {
    background:#1c1b18; color:#7dd3bd;
    border-radius:9px; padding:0.9rem 1.1rem;
    font-family:'JetBrains Mono',monospace; font-size:0.75rem;
    white-space:pre-wrap; max-height:200px; overflow-y:auto;
    line-height:1.6;
}

/* ── Section heading ── */
.sec-head {
    font-size:1.05rem; font-weight:700; color:#28251d;
    margin-bottom:0.6rem; letter-spacing:-0.01em;
}
.sec-sub { font-size:0.82rem; color:#7a7974; margin-top:-4px; margin-bottom:1rem; }

/* ── Warning box ── */
.warn-box {
    background:#fef9ec; border:1px solid #f0d87e;
    border-radius:8px; padding:0.7rem 0.9rem;
    font-size:0.82rem; color:#7a5a00; margin-top:1rem;
}

/* ── Chat tweaks ── */
[data-testid="stChatMessage"] {
    background: #fff !important;
    border: 1px solid rgba(40,37,29,0.07) !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
}
[data-testid="stChatInput"] textarea {
    border-color: rgba(40,37,29,0.2) !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
[data-testid="stTab"] button {
    font-weight:600; font-size:0.88rem;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background:#fff; border:1px solid rgba(40,37,29,0.09);
    border-radius:10px; padding:0.7rem 1rem;
}
</style>
""", unsafe_allow_html=True)


def _bootstrap_env_from_streamlit_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return

    mapping = {
        "AZURE_ENDPOINT": "AZURE_ENDPOINT",
        "AZURE_DEPLOYMENT": "AZURE_DEPLOYMENT",
        "AZURE_API_VERSION": "AZURE_API_VERSION",
        "AZURE_OPENAI_API_KEY": "AZURE_OPENAI_API_KEY",
        "api_key": "api_key",
    }
    for secret_key, env_key in mapping.items():
        value = secrets.get(secret_key)
        if value and not os.environ.get(env_key):
            os.environ[env_key] = str(value)


_bootstrap_env_from_streamlit_secrets()

# ── Imports (after config) ─────────────────────────────────────────────────────
from data_utils  import load_dataset, get_user, get_user_names, build_history_context, week_num
from graph_model import stream_chat_response, run_pattern_analysis, get_graph

# ── Dataset ────────────────────────────────────────────────────────────────────
@st.cache_data
def _load():
    return load_dataset()

dataset    = _load()
user_names = get_user_names(dataset)

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_messages"  not in st.session_state:   st.session_state.chat_messages  = {}  # {user_name: [{role,content}]}
if "pattern_cache"  not in st.session_state:   st.session_state.pattern_cache  = {}  # {user_name: result_dict}
if "graph_ready"    not in st.session_state:
    get_graph()                                                                         # warm up singleton
    st.session_state.graph_ready = True

def get_api_key():
    try:    return st.secrets["AZURE_OPENAI_API_KEY"]
    except: pass
    
    # Check standard name, fallback to local 'api_key' in .env
    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not key:
        key = os.environ.get("api_key", "")
    return key

def thread_id(user_name: str, mode: str) -> str:
    return f"{user_name}_{mode}"

def confidence_badge(conf: str) -> str:
    cls = {"very high":"bvh","high":"bh","medium":"bm","low":"bl"}.get(conf.lower(),"bm")
    return f'<span class="badge {cls}">{conf}</span>'

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Ask<span>First</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Clary · Health Reasoning Engine</div>', unsafe_allow_html=True)
    st.divider()

    selected_user = st.selectbox("👤 Select Patient", user_names, key="selected_user")
    user          = get_user(dataset, selected_user)
    history_ctx   = build_history_context(user)

    # Mini patient card
    st.markdown(f"""
    <div class="sidebar-user">
      <div class="sidebar-user-name">🩺 {user['name']}</div>
      <div class="sidebar-user-meta">
        {user['age']} y/o · {user['gender']} · {user['occupation']}<br>
        {user['location']}<br>
        <strong>Sessions:</strong> {len(user['conversations'])}
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_messages[selected_user] = []
        st.rerun()

    if st.button("🔄 Re-run Pattern Analysis", use_container_width=True):
        if selected_user in st.session_state.pattern_cache:
            del st.session_state.pattern_cache[selected_user]
        st.rerun()

    with st.expander("⚙️ System Specs"):
        st.markdown(f"""
**Memory:** `MemorySaver` stores conversation state per `thread_id` for session persistence.

**LLM:** Azure `gpt-4.1-mini`  
*Reason: Strong JSON adherence, 128k context window, and cost-efficiency.*
        """)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_patterns = st.tabs([
    "💬 Chat with Clary",
    "🔬 Pattern Analysis",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(f'<div class="sec-head">💬 Chat with Clary — {selected_user}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Clary has full memory of the patient\'s health history. Ask anything — Clary will reason temporally across all sessions.</div>', unsafe_allow_html=True)

    # Ensure per-user message store
    if selected_user not in st.session_state.chat_messages:
        st.session_state.chat_messages[selected_user] = []

    msgs = st.session_state.chat_messages[selected_user]

    # Welcome message
    if not msgs:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(f"Hello! I'm **Clary**, your AI health companion. I've analyzed **{selected_user}'s** history — all {len(user['conversations'])} sessions. How can I help today?")

    # Scrollable chat container
    chat_container = st.container(height=500)

    with chat_container:
        # Render history
        for msg in msgs:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "🩺"):
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input(f"Ask Clary about {selected_user}'s health…"):
        if not get_api_key():
            st.error("System API key missing. Please configure it in Streamlit Secrets.")
            st.stop()

        # Add to history
        msgs.append({"role": "user", "content": prompt})
        
        # Rerun to show user message immediately in the container
        st.rerun()

    # If the last message is from the user, trigger the AI response
    if msgs and msgs[-1]["role"] == "user":
        with chat_container:
            with st.chat_message("assistant", avatar="🩺"):
                full_response = st.write_stream(
                    stream_chat_response(
                        user_input      = msgs[-1]["content"],
                        history_context = history_ctx,
                        user_name       = selected_user,
                        thread_id       = thread_id(selected_user, "chat"),
                        api_key         = get_api_key(),
                    )
                )
        msgs.append({"role": "assistant", "content": full_response})
        st.rerun()

    # Suggestion chips if no messages yet
    if not msgs:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            f"What is the most important health pattern in {selected_user}'s history?",
            f"Why does {selected_user} keep getting the same symptom?",
            f"What lifestyle change would have the biggest impact for {selected_user}?",
            f"Are there any connections {selected_user} hasn't noticed?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                st.info(s, icon="💬")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_patterns:
    st.markdown(f'<div class="sec-head">🔬 Pattern Analysis — {selected_user}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Scanning full history for hidden cross-session causal patterns with temporal reasoning.</div>', unsafe_allow_html=True)

    run_btn = st.button(
        f"▶ Analyse {selected_user}'s Full Health History",
        type="primary",
        use_container_width=True,
        disabled=not bool(get_api_key()),
    )

    if not get_api_key():
        st.warning("System API key missing. Please configure it in Streamlit Secrets.")

    # Run or load from cache
    if run_btn and get_api_key():
        with st.spinner(f"🧠 Clary is scanning {selected_user}'s sessions…"):
            result = run_pattern_analysis(
                history_context = history_ctx,
                user_name       = selected_user,
                thread_id       = thread_id(selected_user, "patterns"),
                api_key         = get_api_key(),
            )
        st.session_state.pattern_cache[selected_user] = result
        st.success(f"✅ Found {len(result.get('patterns', []))} patterns!", icon="🔍")

    result = st.session_state.pattern_cache.get(selected_user)

    if result:
        if "error" in result:
            st.error(f"JSON parse error: {result.get('raw', 'Unknown error')}")
        else:
            # ── Reasoning Trace ────────────────────────────────────────────────
            if result.get("reasoning_trace"):
                st.markdown("#### 🧭 Reasoning Trace")
                st.markdown(f'<div class="trace">{result["reasoning_trace"]}</div>', unsafe_allow_html=True)
                st.markdown("")

            patterns = result.get("patterns", [])

            # ── Summary metrics ────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Patterns Found",    len(patterns))
            m2.metric("Very High Conf.",   sum(1 for p in patterns if p.get("confidence","").lower()=="very high"))
            m3.metric("High Confidence",   sum(1 for p in patterns if p.get("confidence","").lower()=="high"))
            m4.metric("Medium / Low",      sum(1 for p in patterns if p.get("confidence","").lower() in ("medium","low")))
            st.markdown("")

            st.markdown(f"#### 📋 {len(patterns)} Pattern{'s' if len(patterns)!=1 else ''} Detected")
            
            with st.expander("📄 View Raw JSON"):
                st.json(result)

            # ── Pattern cards ──────────────────────────────────────────────────
            for p in patterns:
                conf  = p.get("confidence","?")
                badge = confidence_badge(conf)

                sessions_chips = " ".join(f'<span class="chip">{s}</span>' for s in p.get("sessions_involved", []))
                tags_html = " ".join(f'<span class="tag">{t}</span>' for t in p.get("timestamps", []))

                st.markdown(f"""
<div class="pcard">
  <div class="pcard-title">{p.get("pattern_id","?")} &nbsp;·&nbsp; {p.get("title","?")} &nbsp; {badge}</div>

  <div class="label">Sessions Involved</div>
  {sessions_chips}

  <div class="label">Timestamps</div>
  {tags_html}

  <div class="label">Week Range &nbsp;·&nbsp; Lag</div>
  <span style="font-size:0.82rem;color:#28251d">{p.get("week_range","?")} &nbsp;|&nbsp; Lag: {p.get("lag_weeks","?")} weeks</span>

  <div class="label">⏱ Temporal Logic</div>
  <div style="font-size:0.83rem;color:#4a4740;line-height:1.55">{p.get("temporal_logic","")}</div>

  <div class="label">🔗 Causal Narrative</div>
  <div style="font-size:0.85rem;color:#28251d;line-height:1.6">{p.get("causal_narrative","")}</div>

  <div class="label">🎯 Confidence Justification</div>
  <div style="font-size:0.83rem;color:#4a4740;line-height:1.55">{p.get("confidence_justification","")}</div>

  <div class="label">🧪 Intervention Response</div>
  <div style="font-size:0.83rem;color:#4a4740;line-height:1.55">{p.get("intervention_response","")}</div>

  <div class="label">💡 What the Patient Missed</div>
  <div style="font-size:0.83rem;color:#01696f;line-height:1.55;font-style:italic">{p.get("what_user_missed","")}</div>
</div>""", unsafe_allow_html=True)

# Sticky Footer Disclaimer
st.markdown('<div class="footer">⚠️ This is for informational purposes only. Please consult a qualified healthcare provider.</div>', unsafe_allow_html=True)