"""
Embedding engine using sentence-transformers (local, no API key required).
Default model: all-MiniLM-L6-v2  →  384 dimensions.

Groq discontinued their embedding API endpoint; this local approach
gives identical functionality with zero cost and no rate limits.
"""
import numpy as np
from functools import lru_cache
from backend.config import get_settings

settings = get_settings()


class Embedder:
    """Singleton embedding engine backed by sentence-transformers (local)."""

    _instance: "Embedder | None" = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _get_model(self):
        if self._model is None:
            # Import here so startup is fast even if library is missing
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            print(f"[Embedder] Loading model '{settings.embedding_model}' (first run may download weights)...")
            self._model = SentenceTransformer(settings.embedding_model)
            print("[Embedder] Model ready.")
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a FLOAT32 numpy array."""
        model = self._get_model()
        clean_text = text.replace("\n", " ")
        vector = model.encode(clean_text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts in one call."""
        model = self._get_model()
        clean_texts = [t.replace("\n", " ") for t in texts]
        vectors = model.encode(clean_texts, normalize_embeddings=True)
        return [np.array(v, dtype=np.float32) for v in vectors]

    @property
    def dim(self) -> int:
        return settings.embedding_dim


@lru_cache()
def get_embedder() -> Embedder:
    return Embedder()
