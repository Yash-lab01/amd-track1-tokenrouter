import streamlit as st
import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="OptiRoute — Hybrid AI Routing Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #0a0a0f !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1300px !important; }

/* ── Hero header ──────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 0.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin: 0 0 0.75rem;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Tier pills row ───────────────────────────── */
.tier-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin: 1.5rem 0 2.5rem;
    flex-wrap: wrap;
}
.tier-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.tier-pill.t0 { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25); color: #34d399; }
.tier-pill.t1 { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25); color: #60a5fa; }
.tier-pill.t2 { background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.25); color: #a78bfa; }
.tier-dot { width: 7px; height: 7px; border-radius: 50%; }
.t0 .tier-dot { background: #34d399; }
.t1 .tier-dot { background: #60a5fa; }
.t2 .tier-dot { background: #a78bfa; }

/* ── Cards ────────────────────────────────────── */
.card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.75rem;
    height: 100%;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin: 0 0 1.25rem;
}

/* ── Textarea ─────────────────────────────────── */
.stTextArea textarea {
    background: #0d0d14 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.12) !important;
    outline: none !important;
}
.stTextArea label { color: #64748b !important; font-size: 0.82rem !important; }

/* ── Button ───────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Stat blocks ──────────────────────────────── */
.stat-block {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.stat-block-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.35rem;
}
.stat-block-value {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #e2e8f0;
}
.stat-block-sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.2rem;
}

/* ── Route badge ──────────────────────────────── */
.route-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 0.25rem;
}
.route-t0 { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2); color: #34d399; }
.route-t1 { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); color: #60a5fa; }
.route-t2 { background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.2); color: #a78bfa; }

/* ── Response box ─────────────────────────────── */
.response-box {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #7c3aed;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #cbd5e1;
    white-space: pre-wrap;
    word-break: break-word;
    margin-top: 0.5rem;
    max-height: 300px;
    overflow-y: auto;
}

/* ── Architecture section ─────────────────────── */
.arch-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 0.5rem;
}
.arch-card {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
}
.arch-tier-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.arch-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}
.arch-desc { font-size: 0.8rem; color: #64748b; line-height: 1.5; }
.arch-token {
    display: inline-block;
    margin-top: 0.6rem;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
}

/* ── Divider ──────────────────────────────────── */
.divider { border: none; border-top: 1px solid #1e1e2e; margin: 2rem 0; }

/* ── Confidence bar ───────────────────────────── */
.conf-bar-bg {
    background: #1e1e2e;
    border-radius: 999px;
    height: 6px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #60a5fa);
    transition: width 0.4s ease;
}

/* Hide Streamlit elements */
.stTextArea > div > div { background: transparent !important; }
div[data-testid="stVerticalBlock"] > div:has(.card) { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Initialize router ──────────────────────────────────────────────────────────
@st.cache_resource
def get_router():
    from agent.router import HybridRouter
    api_key       = os.environ.get("FIREWORKS_API_KEY", "")
    base_url      = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models = os.environ.get(
        "ALLOWED_MODELS",
        "accounts/fireworks/models/deepseek-v4-pro,accounts/fireworks/models/glm-5p2"
    ).split(",")
    return HybridRouter(api_key, base_url, allowed_models)

try:
    router = get_router()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.info("Ensure the local model is at ./models/gemma-2b-instruct-q4.gguf")
    st.stop()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ AMD Developer Hackathon · Track 1</div>
    <div class="hero-title">OptiRoute</div>
    <p class="hero-sub">3-tier hybrid AI routing — maximize local inference,<br>minimize remote token cost.</p>
</div>

<div class="tier-row">
    <div class="tier-pill t0"><span class="tier-dot"></span>Tier 0 · Direct Solver</div>
    <div class="tier-pill t1"><span class="tier-dot"></span>Tier 1 · Local Gemma-2B</div>
    <div class="tier-pill t2"><span class="tier-dot"></span>Tier 2 · Remote Fireworks AI</div>
</div>
""", unsafe_allow_html=True)


# ── Main two-column layout ─────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Test a Prompt</p>', unsafe_allow_html=True)

    user_prompt = st.text_area(
        label="Enter any query",
        placeholder="Try: 'What is 15% of 240?' or 'Write a Python function to reverse a string' or 'Summarize this article...'",
        height=130,
        label_visibility="collapsed"
    )

    run_btn = st.button("⚡  Route & Generate", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Response output ────────────────────────────────────────────────────────
    if run_btn:
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Routing through tiers…"):
                t_start = time.perf_counter()
                domain, conf = router.classifier.classify(user_prompt)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(router.route_async(user_prompt))
                t_elapsed = time.perf_counter() - t_start

                from agent.evaluator import try_solve_locally
                is_direct = try_solve_locally(domain, user_prompt) is not None

                if is_direct:
                    tier_label = "Tier 0 · Direct Solver"
                    tier_class = "route-t0"
                elif conf >= 0.65:
                    tier_label = "Tier 1 · Local Gemma-2B"
                    tier_class = "route-t1"
                else:
                    tier_label = "Tier 2 · Remote Fireworks AI"
                    tier_class = "route-t2"

                st.session_state.last_run = {
                    "domain": domain,
                    "confidence": conf,
                    "time": t_elapsed,
                    "tier_label": tier_label,
                    "tier_class": tier_class,
                    "result": result,
                }

    if "last_run" in st.session_state:
        r = st.session_state.last_run
        st.markdown(f"""
        <div style="margin-top:1.25rem;">
            <p style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#475569;margin-bottom:0.5rem;">Response</p>
            <div class="response-box">{r['result']}</div>
        </div>
        """, unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Routing Diagnostics</p>', unsafe_allow_html=True)

    if "last_run" in st.session_state:
        r = st.session_state.last_run
        conf_pct = int(r["confidence"] * 100)

        st.markdown(f"""
        <div class="stat-block">
            <div class="stat-block-label">Task Domain</div>
            <div class="stat-block-value">{r["domain"].title()}</div>
            <div class="stat-block-sub">Classifier confidence</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%"></div></div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{conf_pct}%</div>
        </div>

        <div class="stat-block">
            <div class="stat-block-label">Latency</div>
            <div class="stat-block-value">{r["time"]:.2f}s</div>
            <div class="stat-block-sub">End-to-end round-trip</div>
        </div>

        <div class="stat-block">
            <div class="stat-block-label">Execution Route</div>
            <div class="route-badge {r["tier_class"]}">{r["tier_label"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#334155;">
            <div style="font-size:2rem;margin-bottom:0.75rem;">⚡</div>
            <div style="font-size:0.85rem;font-weight:500;">Submit a prompt to see<br>live routing diagnostics.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Architecture section ───────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="card-title" style="margin-bottom:1rem;">Architecture</p>', unsafe_allow_html=True)
st.markdown("""
<div class="arch-row">
    <div class="arch-card">
        <div class="arch-tier-label" style="color:#34d399;">Tier 0</div>
        <div class="arch-title">Direct Solver</div>
        <div class="arch-desc">Math, algebra, and deterministic queries solved instantly via SymPy — no LLM involved.</div>
        <span class="arch-token" style="background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.2);">0 tokens</span>
    </div>
    <div class="arch-card">
        <div class="arch-tier-label" style="color:#60a5fa;">Tier 1</div>
        <div class="arch-title">Local Gemma-2B</div>
        <div class="arch-desc">Quantized Gemma-2B-Instruct runs on CPU. Output is validated programmatically before returning.</div>
        <span class="arch-token" style="background:rgba(59,130,246,0.1);color:#60a5fa;border:1px solid rgba(59,130,246,0.2);">0 remote tokens</span>
    </div>
    <div class="arch-card">
        <div class="arch-tier-label" style="color:#a78bfa;">Tier 2</div>
        <div class="arch-title">Speculative Remote</div>
        <div class="arch-desc">Only if local validation fails — local draft is sent to Fireworks AI for correction, saving ~70% output tokens.</div>
        <span class="arch-token" style="background:rgba(168,85,247,0.1);color:#a78bfa;border:1px solid rgba(168,85,247,0.2);">minimal tokens</span>
    </div>
</div>
""", unsafe_allow_html=True)
