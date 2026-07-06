"""
Redis Vector Memory Store.

Uses Redis Stack's RediSearch module to maintain an HNSW vector index.
Each memory is stored as a Redis HASH with text content, embedding bytes,
and metadata (source, timestamp, tags).
"""
import uuid
import time
import struct
from datetime import datetime, timezone
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from backend.config import get_settings
from backend.embedder import get_embedder

settings = get_settings()
embedder = get_embedder()


class MemoryEntry:
    """Represents a single long-term memory entry."""

    def __init__(
        self,
        memory_id: str,
        text: str,
        source: str,
        score: float = 0.0,
        created_at: str = "",
        tags: str = "",
    ):
        self.memory_id = memory_id
        self.text = text
        self.source = source
        self.score = score
        self.created_at = created_at
        self.tags = tags


class RedisMemoryStore:
    """
    Long-term vector memory store backed by Redis Stack (HNSW index).

    Stores document embeddings and performs sub-millisecond KNN semantic
    search to retrieve relevant context for the AI agent.
    """

    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        self._index_ready = False

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=False,  # We store binary embeddings
            )
        return self._client

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._get_client().ping()
        except Exception:
            return False

    def setup_index(self) -> None:
        """
        Create the HNSW vector index if it doesn't already exist.
        Schema:
          - text      : full text content (TextField)
          - source    : origin label (TextField)
          - tags      : comma-separated tags (TagField)
          - created_at: ISO timestamp string (TextField)
          - ts        : Unix timestamp for ordering (NumericField)
          - embedding : FLOAT32 vector, COSINE distance (VectorField/HNSW)
        """
        client = self._get_client()

        schema = (
            TextField("text", weight=1.0),
            TextField("source"),
            TagField("tags"),
            TextField("created_at"),
            NumericField("ts", sortable=True),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": settings.embedding_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "M": settings.hnsw_m,
                    "EF_CONSTRUCTION": settings.hnsw_ef_construction,
                },
            ),
        )

        try:
            client.ft(settings.memory_index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[settings.memory_key_prefix],
                    index_type=IndexType.HASH,
                ),
            )
            print(f"[Memory] Created HNSW index: {settings.memory_index_name} [DONE]")
        except ResponseError as e:
            if "Index already exists" in str(e):
                print(f"[Memory] Index already exists, reusing [OK]")
            else:
                raise

        self._index_ready = True

    def _ensure_index(self) -> None:
        if not self._index_ready:
            self.setup_index()

    def add_memory(
        self,
        text: str,
        source: str = "manual",
        tags: str = "",
    ) -> str:
        """
        Embed text and store as a memory entry in Redis.

        Returns the generated memory ID.
        """
        self._ensure_index()
        client = self._get_client()

        memory_id = str(uuid.uuid4())
        key = f"{settings.memory_key_prefix}{memory_id}"
        vector = embedder.encode(text)
        now = datetime.now(timezone.utc)

        client.hset(
            key,
            mapping={
                "text": text.encode("utf-8"),
                "source": source.encode("utf-8"),
                "tags": tags.encode("utf-8"),
                "created_at": now.isoformat().encode("utf-8"),
                "ts": int(now.timestamp()),
                "embedding": vector.tobytes(),
            },
        )

        return memory_id

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[MemoryEntry], float]:
        """
        Perform KNN semantic search in the HNSW index.

        Returns (list of MemoryEntry sorted by similarity, latency_ms).
        """
        self._ensure_index()
        client = self._get_client()

        query_vector = embedder.encode(query)

        # Build KNN query using RediSearch vector search dialect 2
        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("text", "source", "tags", "created_at", "vector_score")
            .paging(0, top_k)
            .dialect(2)
        )

        t0 = time.perf_counter()
        results = client.ft(settings.memory_index_name).search(
            q, query_params={"vec": query_vector.tobytes()}
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        entries: list[MemoryEntry] = []
        for doc in results.docs:
            raw_id = doc.id
            memory_id = raw_id.replace(settings.memory_key_prefix, "", 1)

            # score is cosine distance (0 = identical, 2 = opposite)
            # convert to similarity: 1 - distance
            try:
                distance = float(doc.vector_score)
                similarity = round(1.0 - distance, 4)
            except Exception:
                similarity = 0.0

            text_val = doc.text if isinstance(doc.text, str) else doc.text.decode("utf-8", errors="replace")
            source_val = doc.source if isinstance(doc.source, str) else doc.source.decode("utf-8", errors="replace")
            tags_val = doc.tags if isinstance(doc.tags, str) else (doc.tags.decode("utf-8", errors="replace") if doc.tags else "")
            created_at_val = doc.created_at if isinstance(doc.created_at, str) else (doc.created_at.decode("utf-8", errors="replace") if doc.created_at else "")

            entries.append(
                MemoryEntry(
                    memory_id=memory_id,
                    text=text_val,
                    source=source_val,
                    score=similarity,
                    created_at=created_at_val,
                    tags=tags_val,
                )
            )

        return entries, round(latency_ms, 3)

    def get_all_memories(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return all stored memories (no vector search, just scan)."""
        self._ensure_index()
        client = self._get_client()

        # Use FT.SEARCH with wildcard to retrieve all documents
        q = (
            Query("*")
            .return_fields("text", "source", "tags", "created_at", "ts")
            .sort_by("ts", asc=False)
            .paging(0, limit)
            .dialect(2)
        )

        try:
            results = client.ft(settings.memory_index_name).search(q)
        except Exception:
            return []

        memories = []
        for doc in results.docs:
            memory_id = doc.id.replace(settings.memory_key_prefix, "", 1)
            memories.append({
                "id": memory_id,
                "text": doc.text if isinstance(doc.text, str) else doc.text.decode("utf-8", errors="replace"),
                "source": doc.source if isinstance(doc.source, str) else doc.source.decode("utf-8", errors="replace"),
                "tags": doc.tags if isinstance(doc.tags, str) else (doc.tags.decode("utf-8", errors="replace") if doc.tags else ""),
                "created_at": doc.created_at if isinstance(doc.created_at, str) else (doc.created_at.decode("utf-8", errors="replace") if doc.created_at else ""),
            })

        return memories

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID."""
        client = self._get_client()
        key = f"{settings.memory_key_prefix}{memory_id}"
        deleted = client.delete(key)
        return bool(deleted)

    def clear_all(self) -> int:
        """Delete all memories. Returns count of deleted keys."""
        client = self._get_client()
        pattern = f"{settings.memory_key_prefix}*"
        keys = list(client.scan_iter(pattern))
        if keys:
            client.delete(*keys)
        return len(keys)

    def get_stats(self) -> dict[str, Any]:
        """Return index statistics and memory count."""
        self._ensure_index()
        client = self._get_client()

        try:
            info = client.ft(settings.memory_index_name).info()
            # info is a list of alternating key-value pairs
            info_dict: dict = {}
            it = iter(info)
            for k in it:
                try:
                    v = next(it)
                    if isinstance(k, bytes):
                        k = k.decode("utf-8", errors="replace")
                    info_dict[k] = v
                except StopIteration:
                    break

            num_docs = int(info_dict.get("num_docs", 0))
            indexing = info_dict.get("indexing", b"0")
            if isinstance(indexing, bytes):
                indexing = indexing.decode()

            return {
                "memory_count": num_docs,
                "index_name": settings.memory_index_name,
                "embedding_model": settings.embedding_model,
                "embedding_dim": settings.embedding_dim,
                "hnsw_m": settings.hnsw_m,
                "hnsw_ef_construction": settings.hnsw_ef_construction,
                "distance_metric": "COSINE",
                "redis_url": settings.redis_url,
            }
        except Exception as e:
            return {"error": str(e), "memory_count": 0}


# ── Module-level singleton ──────────────────────────────────────────────────
_store: RedisMemoryStore | None = None


def get_memory_store() -> RedisMemoryStore:
    global _store
    if _store is None:
        _store = RedisMemoryStore()
    return _store
