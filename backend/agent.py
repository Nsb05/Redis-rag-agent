"""
AI Agent with RAG-powered long-term memory.

Pipeline:
  1. User sends a message
  2. Embed the query (local, sentence-transformers)
  3. Retrieve top-K semantically similar memories from Redis (HNSW KNN)
  4. Build context prompt: system + memories + session history + user message
  5. Call Groq LLM (llama-3.3-70b-versatile) with the enriched context
  6. Return response + retrieved memories + latency
"""
import re
from collections import deque
from typing import Any

from groq import AsyncGroq

from backend.config import get_settings
from backend.memory import RedisMemoryStore, get_memory_store, MemoryEntry

settings = get_settings()

SYSTEM_PROMPT = """You are an intelligent AI assistant with long-term memory capabilities powered by Redis Vector Search.

When responding:
- You have access to relevant memories retrieved from your long-term memory store via semantic search.
- Use the provided memory context to give accurate, personalized, and contextually rich answers.
- If a memory is directly relevant, reference it naturally in your response.
- If no memories are relevant, respond based on your general knowledge.
- Be concise, helpful, and conversational.
- When asked about your memory system, you can explain how Redis HNSW vector indexing works.

Memory System Details:
- Embedding model: all-MiniLM-L6-v2 (384 dimensions, COSINE similarity)
- Index type: HNSW (Hierarchical Navigable Small World)
- Storage: Redis Stack (sub-millisecond retrieval)
- LLM: Groq (llama-3.3-70b-versatile) — ultra-fast inference
"""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks for better retrieval granularity."""
    words = text.split()
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_size >= len(words):
            break

    return chunks if chunks else [text]


class AIAgent:
    """
    AI Agent with dual memory:
      - Short-term: in-process deque (last N conversation turns per session)
      - Long-term:  Redis vector store (persistent across restarts)
    """

    def __init__(self) -> None:
        self._store: RedisMemoryStore = get_memory_store()
        self._sessions: dict[str, deque] = {}
        self._client: AsyncGroq | None = None

    def _get_groq(self) -> AsyncGroq:
        if self._client is None:
            if not settings.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. "
                    "Get a free key at https://console.groq.com and add it to .env"
                )
            self._client = AsyncGroq(api_key=settings.groq_api_key)
        return self._client

    def _get_session(self, session_id: str) -> deque:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=20)  # last 20 turns
        return self._sessions[session_id]

    async def chat(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """
        Main RAG chat pipeline.

        Returns:
          {
            "response": str,
            "memories_used": list[dict],
            "retrieval_latency_ms": float,
            "session_id": str,
          }
        """
        groq = self._get_groq()
        session = self._get_session(session_id)

        # ── 1. Retrieve relevant long-term memories ────────────────────────
        memories, latency_ms = self._store.search(
            user_message, top_k=settings.max_context_memories
        )

        # ── 2. Build memory context block ──────────────────────────────────
        memory_context = ""
        if memories:
            memory_lines = []
            for i, m in enumerate(memories, 1):
                memory_lines.append(
                    f"  [{i}] (similarity: {m.score:.3f}, source: {m.source})\n"
                    f"      {m.text[:400]}"
                )
            memory_context = "=== Long-Term Memory Context ===\n" + "\n".join(memory_lines)

        # ── 3. Build messages list ─────────────────────────────────────────
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if memory_context:
            messages.append({"role": "system", "content": memory_context})

        # Add conversation history (short-term memory)
        messages.extend(list(session))

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # ── 4. Call Groq LLM ───────────────────────────────────────────────
        completion = await groq.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        response_text = completion.choices[0].message.content

        # ── 5. Update short-term session memory ───────────────────────────
        session.append({"role": "user", "content": user_message})
        session.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "memories_used": [
                {
                    "id": m.memory_id,
                    "text": m.text[:200] + ("..." if len(m.text) > 200 else ""),
                    "source": m.source,
                    "score": m.score,
                }
                for m in memories
                if m.score > 0.1  # only include meaningfully relevant memories
            ],
            "retrieval_latency_ms": latency_ms,
            "session_id": session_id,
        }

    def add_document(
        self,
        text: str,
        source: str = "upload",
        tags: str = "",
    ) -> dict[str, Any]:
        """
        Chunk a document and store each chunk as a long-term memory.

        Returns summary of stored chunks.
        """
        chunks = _chunk_text(
            text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        memory_ids = []
        for chunk in chunks:
            mid = self._store.add_memory(chunk, source=source, tags=tags)
            memory_ids.append(mid)

        return {
            "chunks_stored": len(memory_ids),
            "memory_ids": memory_ids,
            "source": source,
        }

    def add_memory(self, text: str, source: str = "manual", tags: str = "") -> str:
        """Add a single memory entry directly (no chunking)."""
        return self._store.add_memory(text, source=source, tags=tags)

    def clear_session(self, session_id: str) -> None:
        """Clear short-term session history."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()


# ── Module-level singleton ──────────────────────────────────────────────────
_agent: AIAgent | None = None


def get_agent() -> AIAgent:
    global _agent
    if _agent is None:
        _agent = AIAgent()
    return _agent
