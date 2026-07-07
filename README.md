---
title: TokRouter
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: docker
dockerfile: Dockerfile.hf
app_port: 7860
pinned: false
---

# TokRouter — Hybrid Token-Efficient AI Agent
## AMD Developer Hackathon: ACT II — Track 1

An AI agent that autonomously routes tasks between a free local model and a paid remote model (Fireworks AI), minimizing remote token usage while maintaining accuracy above the threshold.

---

## Architecture

```
Task → [Direct Solver] → [Domain Classifier] → [Local Model + Validator] → [Remote (Speculative)] → Answer
           Tier 0               Tier 1                  Tier 2                      Tier 3
         (0 tokens)           (0 tokens)              (0 remote tokens)          (minimal tokens)
```

---

## Setup

### 1. Prerequisites
- Python 3.11+
- Docker Desktop

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Download local model
Download `gemma-2-2b-it-Q4_K_M.gguf` from HuggingFace:
```
https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
```
Place it in: `./models/gemma-2b-instruct-q4.gguf`

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env — add your FIREWORKS_API_KEY
```

### 5. Pre-bake the classifier (one-time)
```bash
python -m agent.classifier
```

---

## Local Testing

### Run the eval harness
```bash
python eval.py
```

### Run against a custom task file
```bash
python main.py data/test_cases.json output/results.json
```

---

## Docker

### Build
```bash
docker build -t tokrouter .
```

### Test locally
```bash
docker run \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/gemma2-9b-it \
  -v "%cd%/data:/input" \
  -v "%cd%/output:/output" \
  tokrouter
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| TF-IDF classifier (not Transformers) | <10MB RAM, <1ms inference, keeps Docker image small |
| llama-cpp-python (not Ollama) | Runs directly in Docker, no separate service needed |
| Gemma-2B GGUF local model | Qualifies for $1,000 Gemma bonus prize |
| Speculative remote correction | Sends local draft, asks remote to verify — saves 70% output tokens |
| Async semaphore | Prevents timeout on large task sets while avoiding CPU thrashing |
| Pre-baked classifier pkl | Zero training cost at container runtime |
| Direct math solver (sympy) | Solves arithmetic/algebra with 0 LLM tokens |
