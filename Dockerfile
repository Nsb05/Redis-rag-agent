# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies needed for torch + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so first startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Railway injects $PORT automatically; default to 8000 locally
ENV PORT=8000

EXPOSE 8000

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
