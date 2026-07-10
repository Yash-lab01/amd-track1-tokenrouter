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
[![Qwen](https://img.shields.io/badge/Qwen-1.5B--Instruct-blue?logo=alibabacloud)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

**An intelligent routing agent that autonomously decides when to use a free local model vs. a paid remote API — minimizing token costs while maintaining accuracy.**

</div>

---

## 🧠 What Is OptiRoute?

OptiRoute is a 3-tier hybrid AI inference system. Instead of blindly sending every request to an expensive remote LLM, it classifies each task and routes it to the cheapest capable tier:

| Tier | Solver | Token Cost | Used For |
|------|--------|-----------|---------|
| **Tier 0** | SymPy Direct Solver | **0 tokens** | Math, arithmetic, algebra |
| **Tier 1** | Local Qwen2.5-1.5B-Instruct (GGUF) | **0 remote tokens** | Sentiment, NER, code, summarization |
| **Tier 2** | Fireworks AI (Remote LLM) | **Minimal tokens** | Complex reasoning and fallback correction |

The result: **up to 80% reduction in remote API token consumption** compared to always using a remote LLM.

---

## 🏗️ Architecture

```text
Prompt
  │
  ▼
[TF-IDF Domain Classifier] ──── detects task type (math / code / NER / etc.)
  │
  ├──▶ [TF-IDF Trap Detector]               (Semantic trap questions forced to remote)
  │
  ├──▶ [Spatial Puzzle Interceptor]         (Logic puzzles forced to remote)
  │
  ├──▶ [Tier 0: SymPy Direct Solver]        (math/algebra → 0 LLM tokens)
  │
  ├──▶ [Zero-Dependency Input Compressor]   (TF-IDF summarizer + Markdown stripper)
  │
  ├──▶ [Tier 1: Local Qwen 1.5B-Instruct]   (Strict GBNF Grammars → 0 remote tokens)
  │         │
  │         └──▶ [Programmatic Validator]   (checks Python syntax, JSON schema, etc.)
  │                    │
  │                    ├── VALID  → Return answer ✅
  │                    └── INVALID → escalate to Tier 2
  │
  └──▶ [Tier 2: Remote Fireworks AI]
            Ranked Fallback Cascade: if primary model 404s, auto-routes to next best model.
```

### Key Components

| File | Purpose |
|------|---------|
| `agent/classifier.py` | TF-IDF domain classifier (< 10MB, < 1ms inference) |
| `agent/trap_detector.py` | Semantic trap detector using TF-IDF vectors to catch tricky logic |
| `agent/compressor.py` | Zero-dependency prompt compressor (TF-IDF + Regex) |
| `agent/local_model.py` | Thread-safe Qwen 1.5B wrapper with GBNF Grammars |
| `agent/remote_model.py`| Remote API client with Ranked Fallback Cascade and Bad-Model Cache |
| `agent/router.py` | Core routing logic, spatial intercepts, and tier escalation |
| `agent/evaluator.py` | Direct solvers and NER schema normalization (Key Aliasing) |
| `streamlit_app.py` | Glassmorphic Streamlit web UI with live diagnostics |
| `main.py` | Entry point (batch eval + Hugging Face Space auto-detection) |
| `eval.py` | Evaluation harness against ground truth (saves trace metadata) |

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

Download the quantized Qwen2.5-1.5B model (~1.1 GB):

```
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Place it at: `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf`

```bash
# Or use wget / curl:
mkdir -p models
wget -O models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
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
LOCAL_MODEL_PATH=./models/qwen2.5-1.5b-instruct-q4_k_m.gguf
ENABLE_TFIDF_TRAP_DETECTOR=1
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
- Execution tier used
- End-to-end latency

### Run the Evaluation Harness

```bash
python eval.py
```

This runs all test cases in `data/` and prints accuracy metrics. It also generates a robust `data/eval_results.json` that includes the full route metadata (model used, why it was chosen).

### Run Batch Inference

```bash
python main.py data/test_cases.json output/results.json
```

---

## 🐳 Docker

### Build the image

```bash
docker build -t optiroute .
```

### Run locally with Docker

**Windows (PowerShell):**
```powershell
docker run `
  -e FIREWORKS_API_KEY=your_key `
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 `
  -e ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-pro `
  -e ENABLE_TFIDF_TRAP_DETECTOR=1 `
  -v "${PWD}/data:/input" `
  -v "${PWD}/output:/output" `
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
| `LOCAL_MODEL_PATH` | No | `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Path to the local GGUF model |
| `ENABLE_TFIDF_TRAP_DETECTOR`| No | `1` | Toggle the semantic trap detector |
| `PORT` | No | `7860` | Port for Streamlit web UI |

---

## 🏆 Phase 2 Hardening & Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Ranked Fallback Cascade** | If a primary remote model throws a 404/403, the router instantly logs it in a bad-model cache and cascades to the next best model in `ALLOWED_MODELS`. Prevents empty responses. |
| **TF-IDF Trap Detector** | Uses a lightweight TF-IDF semantic detector to catch tricky logic prompts that look like factual questions, routing them away from the local model. |
| **NER Schema Normalization** | Aliases LLM hallucinations like `"people"` and `"places"` back to the strict `person` and `location` schema keys, preventing catastrophic validation failures on easy prompts. |
| **GBNF Grammars** | Forces local model to output 100% valid JSON and exact labels, eliminating format-based validator failures. |
| **Zero-Dependency Compressor** | Uses scikit-learn TF-IDF to summarize prompts before API calls, saving 40%+ input tokens without heavy PyTorch bloat. |
| **llama-cpp-python** | Runs directly in Docker using memory-mapped files; no separate container/service needed. |
| **Pre-baked classifier pickle** | Zero training cost at container startup. |

---

## 🔗 Links

- **GitHub:** https://github.com/Yash-lab01/amd-track1-tokenrouter
- **Local Model:** https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
- **Fireworks AI:** https://fireworks.ai
