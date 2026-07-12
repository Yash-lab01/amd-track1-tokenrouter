FROM python:3.11-slim

# ── System deps for llama-cpp-python (CPU build) ──────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Set environment variables ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV LOCAL_MODEL_PATH="/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
ENV PORT=7860
EXPOSE 7860

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# ── Local model handling (conditional download for Hugging Face builds) ────────
# Copy models directory (will copy .gitkeep, and the actual model if present locally)
COPY models/ ./models/

# Download the 1.5B model from Hugging Face hub only if it is not present in the build context
RUN apt-get update && apt-get install -y wget && \
    if [ ! -f /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf ]; then \
        echo "Model not found in build context. Downloading 1.5B model from Hugging Face Hub..."; \
        wget -q -O /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf; \
    else \
        echo "Model found in build context. Skipping download."; \
    fi && \
    apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ── Application source ────────────────────────────────────────────────────────
COPY agent/   ./agent/
COPY data/    ./data/
COPY main.py  .
COPY eval.py  .
COPY streamlit_app.py .

# ── Pre-bake the TF-IDF classifier pickle at build time ───────────────────────
# This removes any training overhead from container startup
RUN python -c "from agent.classifier import DomainClassifier; DomainClassifier()"

# ── I/O directories (expected by eval harness) ───────────────────────────────
RUN mkdir -p /input /output && chmod 777 /input /output

# ── Runtime env defaults (non-sensitive only) ────────────────────────────────
# FIREWORKS_API_KEY, ALLOWED_MODELS are injected at runtime by the eval harness
# Do NOT set them here — keeps the image clean and warning-free
ENV FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
ENV LOCAL_MODEL_PATH="/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# ── Entry point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]