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
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# ── Application source ────────────────────────────────────────────────────────
COPY agent/   ./agent/
COPY data/    ./data/
COPY main.py  .
COPY eval.py  .

# ── Pre-bake the TF-IDF classifier pickle at build time ───────────────────────
# This removes any training overhead from container startup
RUN python -c "from agent.classifier import DomainClassifier; DomainClassifier()"

# ── Local model handling (conditional download for Hugging Face builds) ────────
# Copy models directory (will copy .gitkeep, and the actual model if present locally)
COPY models/ ./models/

# Download the model from Hugging Face hub only if it is not present in the build context
RUN apt-get update && apt-get install -y wget && \
    if [ ! -f /app/models/gemma-2b-instruct-q4.gguf ]; then \
        echo "Model not found in build context. Downloading from Hugging Face Hub..."; \
        wget -q -O /app/models/gemma-2b-instruct-q4.gguf https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf; \
    else \
        echo "Model found in build context. Skipping download."; \
    fi && \
    apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ── I/O directories (expected by eval harness) ───────────────────────────────
RUN mkdir -p /input /output

# ── Environment defaults (real values are injected by the eval harness) ───────
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
ENV ALLOWED_MODELS=""
ENV LOCAL_MODEL_PATH="/app/models/gemma-2b-instruct-q4.gguf"

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "main.py"]
