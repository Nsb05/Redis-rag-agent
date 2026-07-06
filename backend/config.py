"""
Configuration settings loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq (free at console.groq.com)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Embedding model via sentence-transformers (local, no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # HNSW index parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200

    # RAG parameters
    max_context_memories: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Redis index name
    memory_index_name: str = "idx:agent_memory"
    memory_key_prefix: str = "memory:"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
