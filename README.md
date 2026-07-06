# 🧠 AI Agent Long-Term Memory (RAG) with Redis

> **Store document embeddings and perform sub-millisecond semantic search to give AI agents persistent long-term memory.**

Built with **Redis Stack** (HNSW Vector Index) + **sentence-transformers** (local embeddings) + **OpenAI GPT-4o** + **FastAPI**.

---

## Architecture

```
User Query
    │
    ▼
Embed query locally (all-MiniLM-L6-v2, 384 dims)
    │
    ▼
KNN Search in Redis HNSW Index  ←─── sub-millisecond ⚡
    │
    ▼
Top-K memories retrieved (cosine similarity)
    │
    ▼
Inject into GPT-4o context prompt
    │
    ▼
AI Response grounded in long-term memory
```

## Features

- 🧠 **Dual memory** — short-term (session history) + long-term (Redis vector store)
- ⚡ **Sub-millisecond retrieval** — HNSW index with COSINE distance
- 🔒 **Local embeddings** — `all-MiniLM-L6-v2` runs 100% locally, no data sent for embedding
- 📄 **Document ingestion** — auto-chunked, embedded, stored in Redis
- 🔍 **Semantic search** — search memories by meaning, not keywords
- 🎨 **Premium UI** — dark glassmorphism, neural network canvas, real-time stats

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.10+   |
| Docker      | Latest  |
| OpenAI API Key | GPT-4o access |

---

## Quick Start

### 1. Clone & enter the project
```bash
cd redis-rag-agent
```

### 2. Set up your API key
```bash
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

### 3. Start Redis Stack
```bash
docker-compose up -d
```
Redis will be available on `localhost:6379`.
RedisInsight GUI will be at http://localhost:8001.

### 4. Create a Python virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First run downloads the `all-MiniLM-L6-v2` model (~90MB). This is one-time only.

### 6. Start the backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open the UI
Open http://localhost:8000 in your browser (FastAPI serves the frontend).

Or open `frontend/index.html` directly in your browser.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Redis + API health check |
| `POST` | `/chat` | Chat with the AI agent (RAG) |
| `POST` | `/memories` | Add a single memory |
| `POST` | `/upload` | Upload + chunk a document |
| `GET` | `/memories` | List all memories |
| `DELETE` | `/memories/{id}` | Delete a specific memory |
| `DELETE` | `/memories` | Clear ALL memories |
| `GET` | `/search?q=...` | Raw semantic search |
| `GET` | `/stats` | Redis index statistics |

Interactive docs: http://localhost:8000/docs

---

## Configuration

Edit `.env` to customize:

```env
OPENAI_API_KEY=sk-...          # Required for GPT-4o
REDIS_URL=redis://localhost:6379
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
HNSW_M=16                      # HNSW graph connections
HNSW_EF_CONSTRUCTION=200       # Build quality
MAX_CONTEXT_MEMORIES=5         # Memories injected per query
CHUNK_SIZE=500                 # Words per chunk
CHUNK_OVERLAP=50               # Overlap between chunks
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vector DB | Redis Stack (RediSearch) |
| Index | HNSW, COSINE distance |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| LLM | OpenAI GPT-4o |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + Vanilla CSS + JS |

---

## How HNSW Works

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where each node (memory vector) is connected to its nearest neighbors. During search:

1. Entry point at the top layer
2. Greedily navigate towards the query vector
3. Descend through layers, refining neighbors
4. Return top-K approximate nearest neighbors

**Result**: O(log N) search complexity with sub-millisecond latency even at millions of vectors.
