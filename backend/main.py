"""
FastAPI application — AI Agent Long-Term Memory (RAG) API.

Endpoints:
  POST   /chat                — chat with the AI agent (RAG-powered)
  POST   /memories            — manually add a memory
  POST   /upload              — upload a document (chunked + embedded)
  GET    /memories            — list all stored memories
  DELETE /memories/{id}       — delete a specific memory
  DELETE /memories            — clear ALL memories
  GET    /search              — raw semantic search
  GET    /stats               — Redis index stats
  GET    /health              — connectivity health check
"""
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from backend.config import get_settings
from backend.memory import get_memory_store
from backend.agent import get_agent

settings = get_settings()

app = FastAPI(
    title="AI Agent Long-Term Memory (RAG)",
    description="Redis Vector DB + Groq LLaMA-3.3 + sentence-transformers",
    version="1.0.0",
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    memories_used: list[dict]
    retrieval_latency_ms: float
    session_id: str


class AddMemoryRequest(BaseModel):
    text: str
    source: str = "manual"
    tags: str = ""


class MemoryResponse(BaseModel):
    id: str
    text: str
    source: str
    tags: str
    created_at: str


class SearchResult(BaseModel):
    id: str
    text: str
    source: str
    score: float
    created_at: str


# ── Startup: initialise Redis index ────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    store = get_memory_store()
    if store.ping():
        store.setup_index()
        print("[API] Redis connected and index ready [OK]")
    else:
        print("[API] [WARNING] Redis not reachable — start Redis Stack with docker compose up -d")


# ── Health check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict:
    store = get_memory_store()
    redis_ok = store.ping()
    # Always return HTTP 200 — Railway healthcheck only checks status code.
    # Redis connectivity is reported in the body for observability.
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "unreachable",
        "embedding_model": settings.embedding_model,
        "llm": settings.groq_model,
        "groq_key_set": bool(settings.groq_api_key),
    }


# ── Chat endpoint ───────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Send a message to the AI agent.
    The agent retrieves relevant long-term memories from Redis and uses them
    as context for GPT-4o to generate a response.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    store = get_memory_store()
    if not store.ping():
        raise HTTPException(
            status_code=503,
            detail="Redis is not reachable. Start it with: docker-compose up -d",
        )

    try:
        agent = get_agent()
        result = await agent.chat(req.message, session_id=req.session_id)
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ── Memory endpoints ────────────────────────────────────────────────────────

@app.post("/memories")
async def add_memory(req: AddMemoryRequest) -> dict:
    """Manually add a single memory to the long-term store."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Memory text cannot be empty.")

    store = get_memory_store()
    if not store.ping():
        raise HTTPException(status_code=503, detail="Redis not reachable.")

    memory_id = store.add_memory(req.text.strip(), source=req.source, tags=req.tags)
    return {"id": memory_id, "status": "stored"}


@app.get("/memories")
async def list_memories(limit: int = Query(default=100, le=500)) -> dict:
    """List all stored memories (most recent first)."""
    store = get_memory_store()
    if not store.ping():
        raise HTTPException(status_code=503, detail="Redis not reachable.")

    memories = store.get_all_memories(limit=limit)
    return {"memories": memories, "count": len(memories)}


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """Delete a specific memory by ID."""
    store = get_memory_store()
    deleted = store.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")
    return {"deleted": memory_id, "status": "ok"}


@app.delete("/memories")
async def clear_all_memories() -> dict:
    """Delete ALL memories from the vector store."""
    store = get_memory_store()
    count = store.clear_all()
    return {"deleted_count": count, "status": "cleared"}


# ── Document upload endpoint ────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(
    text: str = Form(default=""),
    source: str = Form(default="upload"),
    tags: str = Form(default=""),
    file: UploadFile | None = File(default=None),
) -> dict:
    """
    Upload a document for ingestion into long-term memory.
    Accepts either raw text (via form field) or a .txt file upload.
    The document is auto-chunked and each chunk is embedded + stored in Redis.
    """
    store = get_memory_store()
    if not store.ping():
        raise HTTPException(status_code=503, detail="Redis not reachable.")

    # Resolve content
    content = ""
    if file is not None:
        raw = await file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("latin-1")
        source = source or file.filename or "file_upload"
    elif text.strip():
        content = text.strip()
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a 'text' form field or a file upload.",
        )

    agent = get_agent()
    result = agent.add_document(content, source=source, tags=tags)
    return result


# ── Semantic search endpoint ────────────────────────────────────────────────

@app.get("/search")
async def semantic_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, le=20),
) -> dict:
    """
    Perform raw semantic search in the Redis vector index.
    Returns documents ranked by cosine similarity.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    store = get_memory_store()
    if not store.ping():
        raise HTTPException(status_code=503, detail="Redis not reachable.")

    results, latency_ms = store.search(q, top_k=top_k)

    return {
        "query": q,
        "results": [
            {
                "id": r.memory_id,
                "text": r.text,
                "source": r.source,
                "score": r.score,
                "created_at": r.created_at,
            }
            for r in results
        ],
        "latency_ms": latency_ms,
        "count": len(results),
    }


# ── Stats endpoint ──────────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats() -> dict:
    """Return Redis index statistics and configuration."""
    store = get_memory_store()
    if not store.ping():
        raise HTTPException(status_code=503, detail="Redis not reachable.")
    return store.get_stats()


# ── Serve frontend ──────────────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
