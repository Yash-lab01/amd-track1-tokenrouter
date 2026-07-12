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

# ⚡ OptiRoute — Remote-First Accuracy Router

### AMD Developer Hackathon: ACT II — Track 1

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![Fireworks AI](https://img.shields.io/badge/Fireworks%20AI-integrated-orange)](https://fireworks.ai)
[![Qwen](https://img.shields.io/badge/Qwen-1.5B--Instruct-blue?logo=alibabacloud)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

**An intelligent routing agent that uses deterministic solvers and remote LLMs with chain-of-thought, self-consistency voting, and judge-aware prompts to maximize accuracy.**

</div>

---

## 🧠 What Is OptiRoute?

OptiRoute is a **remote-first accuracy router** with exact local shortcuts. It classifies each task, tries deterministic solvers first (0 tokens), then routes to the best remote model with domain-specific prompts, chain-of-thought reasoning, and self-consistency voting.

| Tier         | Solver                                       | Token Cost          | Used For                                 |
| ------------ | -------------------------------------------- | ------------------- | ---------------------------------------- |
| **Tier 0**   | Deterministic Solvers (Python)               | **0 tokens**        | Math, facts, sentiment rules, code fixes |
| **Tier 1**   | Remote Fireworks AI (CoT + Self-Consistency) | Remote tokens       | Everything Tier 0 can't solve            |
| **Fallback** | Local Qwen 1.5B (Emergency)                  | **0 remote tokens** | Only if remote completely fails          |

---

## 🏗️ Architecture

```text
Prompt
  │
  ▼
[Cache Check] ──── Duplicate? → Return cached answer
  │
  ▼
[TF-IDF Domain Classifier] ──── detects task type (8 domains)
  │
  ├──▶ [Trap Detector] ────────── Spatial puzzles & logic traps → force remote
  │
  ├──▶ [Tier 0: Deterministic Solver]
  │        ├── Math patterns (SymPy, regex) → 0 tokens
  │        ├── Fact table (59 stable facts) → 0 tokens
  │        ├── Sentiment rules (lexicon) → 0 tokens
  │        └── Code fixes (AST patterns) → 0 tokens
  │
  ├──▶ [Tier 1: Remote Fireworks AI]
  │        ├── Model specialization (Kimi for code, Minimax for logic, Gemma for factual)
  │        ├── Chain-of-Thought prompts with "Final Answer: X" extraction
  │        ├── Self-consistency voting (3 parallel calls for math/logic)
  │        ├── Few-shot examples in system prompts
  │        ├── Judge-aware prompts (sentiment needs reason, factual needs explanation)
  │        ├── Dynamic max_tokens based on prompt complexity
  │        └── Ranked fallback cascade (bad-model cache)
  │
  ├──▶ [Validate + Postprocess]
  │        ├── NER: JSON schema validation + key aliasing
  │        ├── Code: AST syntax check
  │        ├── Sentiment: Label + reason format
  │        ├── Math: CoT answer extraction
  │        └── Logic: CoT answer extraction + yes/no/impossible handling
  │
  └──▶ [Retry on malformed] ──── NER/code/logic only, one retry
         │
         ▼
    [Emergency Local Fallback] ── Only if remote completely fails
```

### Key Components

| File                     | Purpose                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `agent/classifier.py`    | TF-IDF domain classifier (< 10MB, < 1ms inference)                           |
| `agent/trap_detector.py` | Semantic trap detector for logic puzzles                                     |
| `agent/compressor.py`    | Safe prompt compressor (preserves reasoning prompts)                         |
| `agent/local_model.py`   | Thread-safe Qwen 1.5B wrapper (emergency fallback)                           |
| `agent/remote_model.py`  | Remote API client with CoT, self-consistency, few-shot, model specialization |
| `agent/router.py`        | Core routing logic — remote-first with deterministic shortcuts               |
| `agent/evaluator.py`     | Deterministic solvers, validators, judge-aware postprocessing                |
| `streamlit_app.py`       | Streamlit web UI with live diagnostics                                       |
| `main.py`                | Entry point (batch eval + Hugging Face Space auto-detection)                 |
| `eval.py`                | Evaluation harness against ground truth                                      |

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.11+
- Docker Desktop (optional, for containerized runs)
- A [Fireworks AI](https://fireworks.ai) API key

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

> **Note for llama-cpp-python:** Use the prebuilt CPU wheel:
>
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

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/gemma-4-31b-it,accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/gemma-4-31b-it-nvfp4
LOCAL_MODEL_PATH=./models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

## 🖥️ Usage

### Run the Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

### Run the Evaluation Harness

```bash
python eval.py data/dev_eval.json
```

### Run Batch Inference

```bash
python main.py data/tasks.json output/results.json
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
  -e ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/gemma-4-26b-a4b-it `
  -v "${PWD}/data:/input" `
  -v "${PWD}/output:/output" `
  optiroute
```

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

| Variable                     | Required | Default                                      | Description                       |
| ---------------------------- | -------- | -------------------------------------------- | --------------------------------- |
| `FIREWORKS_API_KEY`          | ✅ Yes   | —                                            | Your Fireworks AI API key         |
| `FIREWORKS_BASE_URL`         | No       | `https://api.fireworks.ai/inference/v1`      | Fireworks endpoint                |
| `ALLOWED_MODELS`             | No       | 5 Fireworks models                           | Comma-separated model IDs         |
| `LOCAL_MODEL_PATH`           | No       | `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Path to the local GGUF model      |
| `ENABLE_TFIDF_TRAP_DETECTOR` | No       | `0`                                          | Toggle the semantic trap detector |
| `PORT`                       | No       | `7860`                                       | Port for Streamlit web UI         |

---

## 🏆 Key Design Decisions

| Decision                      | Rationale                                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Remote-First Architecture** | Local 1.5B model hallucinates on hidden test questions. Remote models are the accuracy engine.                                             |
| **Chain-of-Thought Prompts**  | Models reason much better with step-by-step thinking. We extract only the final answer.                                                    |
| **Self-Consistency Voting**   | 3 parallel calls for math/logic with majority vote. Proven accuracy boost.                                                                 |
| **Judge-Aware Prompts**       | The judge is an LLM checking semantic completeness. Sentiment needs a reason, factual needs explanation, summarization needs exact format. |
| **Deterministic Solvers**     | 59 stable facts + 25+ math patterns solved with 0 tokens and 100% accuracy.                                                                |
| **Model Specialization**      | Kimi for code, Minimax for logic, Gemma 26B for factual/NER/sentiment.                                                                     |
| **Ranked Fallback Cascade**   | If a model 404s, instantly cascade to next best. Bad-model cache prevents retries.                                                         |
| **Safe Prompt Compression**   | Preserves full prompts for logic/math/NER. Only cleans whitespace/markdown for other domains.                                              |
| **Emergency Local Fallback**  | If all remote models fail, local Qwen 1.5B generates a best-effort answer. Better than empty string.                                       |

---

## 📊 Domain Routing Strategy

| Domain            | Tier 0 (0 tokens)             | Tier 1 (Remote)                  | Model Preference    |
| ----------------- | ----------------------------- | -------------------------------- | ------------------- |
| **Math**          | 25+ deterministic patterns    | CoT + self-consistency (3 calls) | Gemma 26B → Minimax |
| **Factual**       | 59 stable facts               | Complete answer with explanation | Gemma 26B → Minimax |
| **Sentiment**     | Lexicon rules (obvious cases) | Label + one-sentence reason      | Gemma 26B           |
| **Logic**         | —                             | CoT + self-consistency (3 calls) | Minimax → Gemma 31B |
| **NER**           | —                             | JSON with all entities           | Gemma 26B           |
| **Debugging**     | AST pattern fixes             | Corrected code only              | Kimi → Gemma 26B    |
| **Codegen**       | AST pattern fixes             | Working code only                | Kimi → Gemma 26B    |
| **Summarization** | —                             | Exact format (sentences/bullets) | Gemma 26B           |

---

## 🔗 Links

- **GitHub:** https://github.com/Yash-lab01/amd-track1-tokenrouter
- **HuggingFace Space:** https://huggingface.co/spaces/YashB-21/amd-track1-tokenrouter
- **Local Model:** https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
- **Fireworks AI:** https://fireworks.ai
- **Full Hackathon Details:** See `HACKATHON_FULL_DETAILS.md`