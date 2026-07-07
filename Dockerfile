FROM python:3.11-slim

# ── System deps for llama-cpp-python (CPU build) ──────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY agent/   ./agent/
COPY data/    ./data/
COPY main.py  .
COPY eval.py  .

# ── Pre-bake the TF-IDF classifier pickle at build time ───────────────────────
# This removes any training overhead from container startup
RUN python -c "from agent.classifier import DomainClassifier; DomainClassifier()"

# ── Pre-download local model (comment out and COPY instead for faster builds) ─
# Uncomment the line below ONLY if you have the model in ./models/ locally
COPY models/ ./models/

# ── I/O directories (expected by eval harness) ───────────────────────────────
RUN mkdir -p /input /output

# ── Environment defaults (real values are injected by the eval harness) ───────
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
ENV ALLOWED_MODELS=""
ENV LOCAL_MODEL_PATH="/app/models/gemma-2b-instruct-q4.gguf"

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "main.py"]
