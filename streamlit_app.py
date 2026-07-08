import streamlit as st
import asyncio
import os
import time
from dotenv import load_dotenv

# Load local env if present
load_dotenv()

# Set page config with premium theme settings
st.set_page_config(
    page_title="OptiRoute — Hybrid AI Routing Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium dark glassmorphism styling
st.markdown("""
    <style>
        /* Base styles */
        .stApp {
            background: linear-gradient(135deg, #0f0c20 0%, #15102a 50%, #06040a 100%);
            color: #e2e8f0;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        
        /* Glassmorphism containers */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px rgba(255, 255, 255, 0.08) solid;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Headers & gradients */
        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        .gradient-text {
            background: linear-gradient(90deg, #a855f7 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        /* Stats dashboard style */
        .stat-val {
            font-size: 2rem;
            font-weight: 800;
            color: #3b82f6;
            margin-bottom: 4px;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Inputs & Buttons */
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 8px !important;
        }
        .stButton button {
            background: linear-gradient(90deg, #a855f7 0%, #3b82f6 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 10px 24px !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 20px rgba(168, 85, 247, 0.4) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize router
@st.cache_resource
def get_router():
    from agent.router import HybridRouter
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models = os.environ.get("ALLOWED_MODELS", "accounts/fireworks/models/deepseek-v4-pro,accounts/fireworks/models/glm-5p2").split(",")
    return HybridRouter(api_key, base_url, allowed_models)

try:
    router = get_router()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Make sure you downloaded the local model to ./models/gemma-2b-instruct-q4.gguf")
    st.stop()

# Header layout
st.markdown("<h1 style='text-align: center;'><span class='gradient-text'>⚡ OptiRoute</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 40px;'>3-Tier Hybrid AI Routing Agent optimized for zero-token local inference & speculative correction.</p>", unsafe_allow_html=True)

# Main UI layout
col_input, col_stats = st.columns([2, 1])

with col_input:
    st.markdown('<div class="glass-card"><h3>Test a Prompt</h3>', unsafe_allow_html=True)
    user_prompt = st.text_area("Enter any query to route:", placeholder="Type a programming bug, a math equation, or a factual question...", height=120)
    
    if st.button("Route & Generate"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Routing prompt through tiers..."):
                t_start = time.perf_counter()
                
                # Classify first to show visual feedback
                domain, conf = router.classifier.classify(user_prompt)
                
                # Full route run
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(router.route_async(user_prompt))
                
                t_elapsed = time.perf_counter() - t_start
                
                # Determine routing tier used
                import sympy
                from agent.evaluator import try_solve_locally
                
                is_direct = try_solve_locally(domain, user_prompt) is not None
                
                # Display output
                st.markdown("<h4>Agent Response</h4>", unsafe_allow_html=True)
                st.write(result)
                
                # Save execution details in session state for stats display
                st.session_state.last_run = {
                    "domain": domain,
                    "confidence": conf,
                    "time": t_elapsed,
                    "tier": "Tier 0: Direct Solver (0 tokens)" if is_direct 
                            else ("Tier 1/2: Local Gemma-2B (0 API tokens)" if "VALID" not in result 
                                  else "Tier 3: Speculative Remote Correction (Paid)")
                }
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats:
    st.markdown('<div class="glass-card"><h3>Routing Diagnostics</h3>', unsafe_allow_html=True)
    
    if 'last_run' in st.session_state:
        run = st.session_state.last_run
        
        # Domain Pill
        st.markdown(f"<div class='stat-label'>Detected Task Domain</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-val' style='color: #a855f7;'>{run['domain'].upper()}</div>", unsafe_allow_html=True)
        st.write(f"Classifier Confidence: **{run['confidence']:.2f}**")
        st.markdown("---")
        
        # Time taken
        st.markdown(f"<div class='stat-label'>Latency</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-val'>{run['time']:.2f}s</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Route selection
        st.markdown(f"<div class='stat-label'>Execution Route</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight: bold; margin-top: 8px; color: #10b981;'>{run['tier']}</div>", unsafe_allow_html=True)
        
    else:
        st.info("Run a prompt to see classification and routing latency diagnostics.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Architecture explanation section
st.markdown("""
    <div class="glass-card">
        <h3>How it Works</h3>
        <p>OptiRoute implements a cascading 3-tier routing mechanism to balance speed, cost, and accuracy:</p>
        <ol>
            <li><b>Tier 0 (Direct Solve):</b> Math and simple deterministic queries are solved instantly via python execution (0 LLM tokens).</li>
            <li><b>Tier 1/2 (Local Gemma-2B):</b> Prompts are routed to a quantized local Gemma model. Programmatic validation checks the syntax (e.g. valid Python, correct JSON keys). If valid, it returns the result (0 remote tokens).</li>
            <li><b>Tier 3 (Speculative Remote):</b> If local validation fails, the query and the local draft are sent to Fireworks AI. The remote model validates or corrects the draft, saving output token consumption.</li>
        </ol>
    </div>
""", unsafe_allow_html=True)
