# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── CPU-only PyTorch 2.5.1 (compatible with sentence-transformers 3.3.1) ─────
# Must install before requirements.txt so pip doesn't pull in the GPU build.
RUN pip install --no-cache-dir \
    "torch==2.5.1+cpu" \
    "torchvision==0.20.1+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# ── Remaining Python dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pin cache dirs so build-time model is found at runtime ────────────────────
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Pre-download the model at build time (weights baked into the image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Railway injects $PORT; default to 8000 locally
ENV PORT=8000

EXPOSE 8000

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
