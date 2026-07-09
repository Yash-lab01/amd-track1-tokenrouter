---
title: OptiRoute
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: docker
dockerfile: Dockerfile
app_port: 7860
pinned: false
---

<div align="center">

# ⚡ OptiRoute — Hybrid Token-Efficient AI Agent

### AMD Developer Hackathon: ACT II — Track 1

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![Fireworks AI](https://img.shields.io/badge/Fireworks%20AI-integrated-orange)](https://fireworks.ai)
[![Gemma](https://img.shields.io/badge/Gemma-2B--Instruct-green?logo=google)](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF)

**An intelligent routing agent that autonomously decides when to use a free local model vs. a paid remote API — minimizing token costs while maintaining accuracy.**

</div>

---

## 🧠 What Is OptiRoute?

OptiRoute is a 3-tier hybrid AI inference system. Instead of blindly sending every request to an expensive remote LLM, it classifies each task and routes it to the cheapest capable tier:

| Tier | Solver | Token Cost | Used For |
|------|--------|-----------|---------|
| **Tier 0** | SymPy Direct Solver | **0 tokens** | Math, arithmetic, algebra |
| **Tier 1/2** | Local Gemma-2B-Instruct (GGUF) | **0 remote tokens** | Sentiment, NER, code, summarization |
| **Tier 3** | Fireworks AI (Remote LLM) | **Minimal tokens** | Speculative correction only when local fails |

The result: **up to 80% reduction in remote API token consumption** compared to always using a remote LLM.

---

## 🏗️ Architecture

```text
Prompt
  │
  ▼
[TF-IDF Domain Classifier] ──── detects task type (math / code / NER / etc.)
  │
  ├──▶ [Spatial Puzzle Interceptor]         (Logic puzzles forced to remote)
  │
  ├──▶ [Tier 0: SymPy Direct Solver]        (math/algebra → 0 LLM tokens)
  │
  ├──▶ [Zero-Dependency Input Compressor]   (TF-IDF summarizer + Markdown stripper)
  │
  ├──▶ [Tier 1/2: Local Gemma-2B-Instruct]  (Strict GBNF Grammars → 0 remote tokens)
  │         │
  │         ├──▶ [Math Self-Consistency]    (Two-pass verification for math)
  │         │
  │         └──▶ [Programmatic Validator]   (checks Python syntax, JSON keys, etc.)
  │                    │
  │                    ├── VALID  → Return answer ✅
  │                    └── INVALID → escalate to Tier 3
  │
  └──▶ [Tier 3: The Lean Auditor]           (Remote Fireworks AI)
            Audits local answers ("Approve or Replace") → saves ~90% output tokens
```

### Key Components

| File | Purpose |
|------|---------|
| `agent/classifier.py` | TF-IDF domain classifier (< 10MB, < 1ms inference) |
| `agent/compressor.py` | Zero-dependency prompt compressor (TF-IDF + Regex) |
| `agent/local_model.py` | Thread-safe Gemma-2B wrapper with GBNF Grammars |
| `agent/remote_model.py`| Remote API client with the "Approve or Replace" Auditor |
| `agent/router.py` | Core routing logic, spatial intercepts, and tier escalation |
| `agent/evaluator.py` | Direct solvers (SymPy math, regex, etc.) |
| `agent/validator.py` | Output validation per domain |
| `streamlit_app.py` | Glassmorphic Streamlit web UI with live diagnostics |
| `main.py` | Entry point (batch eval + Hugging Face Space auto-detection) |
| `eval.py` | Evaluation harness against ground truth |
| `data/router_training.json` | 168-example training corpus for classifier |

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.11+
- Docker Desktop (optional, for containerized runs)
- A [Fireworks AI](https://fireworks.ai) API key (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/Yash-lab01/amd-track1-tokenrouter.git
cd amd-track1-tokenrouter
```

### 2. Set up a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for llama-cpp-python:** If the above is slow (it compiles C++ from source), use the prebuilt CPU wheel:
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
> pip install -r requirements.txt
> ```

### 4. Download the local model

Download the quantized Gemma-2B model (~1.7 GB):

```
https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf
```

Place it at: `./models/gemma-2b-instruct-q4.gguf`

```bash
# Or use wget / curl:
mkdir -p models
wget -O models/gemma-2b-instruct-q4.gguf \
  https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-pro,accounts/fireworks/models/glm-5p2
LOCAL_MODEL_PATH=./models/gemma-2b-instruct-q4.gguf
```

---

## 🖥️ Usage

### Run the Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser. The UI shows:
- Live prompt routing
- Detected task domain + classifier confidence score
- Execution tier used (Tier 0 / 1 / 2 / 3)
- End-to-end latency

### Run the Evaluation Harness

```bash
python eval.py
```

This runs all test cases in `data/` and prints accuracy/token savings metrics.

### Run Batch Inference

```bash
python main.py data/test_cases.json output/results.json
```

Input format (`tasks.json`):
```json
[
  {"task_id": "1", "prompt": "What is the capital of France?"},
  {"task_id": "2", "prompt": "Solve: 3x + 5 = 20"},
  {"task_id": "3", "prompt": "Write a Python function to reverse a string"}
]
```

Output format (`results.json`):
```json
[
  {"task_id": "1", "answer": "Paris"},
  {"task_id": "2", "answer": "Answer: 5"},
  {"task_id": "3", "answer": "```python\ndef reverse_string(s): return s[::-1]\n```"}
]
```

---

## 🐳 Docker

### Build the image

```bash
docker build -t optiroute .
```

> **Note:** The Dockerfile uses a prebuilt `llama-cpp-python` CPU wheel, so the build completes in ~2 minutes (no C++ compilation).

### Run locally with Docker

**Windows (PowerShell):**
```powershell
docker run `
  -e FIREWORKS_API_KEY=your_key `
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 `
  -e ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-pro `
  -v "${PWD}/data:/input" `
  -v "${PWD}/output:/output" `
  optiroute
```

**macOS / Linux:**
```bash
docker run \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-pro \
  -v "$(pwd)/data:/input" \
  -v "$(pwd)/output:/output" \
  optiroute
```

### Run the Streamlit UI inside Docker

```bash
docker run -p 7860:7860 \
  -e SPACE_ID=local \
  -e FIREWORKS_API_KEY=your_key \
  -e ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-pro \
  optiroute
```

Then open [http://localhost:7860](http://localhost:7860).

---

## 🌐 Deployment

### Hugging Face Spaces (Live Demo)

The project is deployed at:
> **https://huggingface.co/spaces/YashB-21/amd-track1-tokenrouter**

To deploy your own copy:
1. Fork this repo to a new Hugging Face Space (SDK: Docker)
2. Set the following **Secrets** in your Space settings:
   - `FIREWORKS_API_KEY` — your Fireworks AI API key
   - `ALLOWED_MODELS` — comma-separated model IDs
3. Push to the Space — the Dockerfile handles everything automatically

---

## ⚙️ Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FIREWORKS_API_KEY` | ✅ Yes | — | Your Fireworks AI API key |
| `FIREWORKS_BASE_URL` | No | `https://api.fireworks.ai/inference/v1` | Fireworks endpoint |
| `ALLOWED_MODELS` | No | `accounts/fireworks/models/deepseek-v4-pro` | Comma-separated allowed model IDs |
| `LOCAL_MODEL_PATH` | No | `./models/gemma-2b-instruct-q4.gguf` | Path to the local GGUF model |
| `PORT` | No | `7860` | Port for Streamlit web UI |

---

## 🏆 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **The Lean Auditor** | Remote model audits local answers instead of generating from scratch. Saves ~90% of Fireworks output tokens per task. |
| **GBNF Grammars** | Forces local model to output 100% valid JSON and exact labels, eliminating format-based validator failures. |
| **Zero-Dependency Compressor** | Uses scikit-learn TF-IDF to summarize prompts before API calls, saving 40%+ input tokens without heavy PyTorch bloat. |
| **Math Self-Consistency** | Double-checks local math answers via multi-pass sampling to catch silent hallucinations before they are returned. |
| **Spatial Interceptor** | Pre-emptively routes spatial/positional puzzles to remote models, bypassing the local model's known weaknesses. |
| **TF-IDF classifier** | < 10MB RAM, < 1ms inference, keeps Docker image small |
| **llama-cpp-python** | Runs directly in Docker, no separate service needed |
| **Gemma-2B GGUF** local model | Qualifies for Gemma bonus prize, Apache 2.0 license |
| **Async semaphore** | Prevents timeout on large task sets while avoiding CPU thrashing |
| **Pre-baked classifier pickle** | Zero training cost at container startup |
| **Direct SymPy solver** | Math/algebra solved with 0 LLM tokens at all |

---

## 📊 Performance Profile

| Task Domain | Tier Used | Remote Tokens |
|-------------|-----------|--------------|
| Math / Arithmetic | Tier 0 (SymPy) | **0** |
| Sentiment Analysis | Tier 1/2 (Local) | **0** |
| Named Entity Recognition | Tier 1/2 (Local) | **0** |
| Summarization | Tier 1/2 (Local) | **0** |
| Code Generation | Tier 1/2 (Local) | **0** (if valid) |
| Debugging | Tier 1/2 → 3 | **Minimal** (draft sent) |
| Complex Logic | Tier 3 | Standard |

---

## 📁 Project Structure

```
amd-track1-tokenrouter/
├── agent/
│   ├── classifier.py       # TF-IDF domain router
│   ├── evaluator.py        # Direct solvers (SymPy, etc.)
│   ├── local_model.py      # Gemma-2B wrapper (llama-cpp-python)
│   ├── router.py           # HybridRouter — core routing logic
│   └── validator.py        # Per-domain output validation
├── data/
│   ├── router_training.json # 168-example classifier training corpus
│   └── test_cases.json      # Sample eval task set
├── models/
│   └── .gitkeep            # Placeholder (model file excluded from git)
├── streamlit_app.py        # Premium glassmorphic web UI
├── main.py                 # Entry point + HF Space auto-detection
├── eval.py                 # Evaluation harness
├── Dockerfile              # Unified Docker build (local + HF Spaces)
├── requirements.txt        # Python dependencies
└── .env.example            # Example environment file
```

---

## 🔗 Links

- **Live Demo:** https://huggingface.co/spaces/YashB-21/amd-track1-tokenrouter
- **GitHub:** https://github.com/Yash-lab01/amd-track1-tokenrouter
- **Local Model:** https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
- **Fireworks AI:** https://fireworks.ai
