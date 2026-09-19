"""Local embedding backend.

Primary path: sentence-transformers (fully local, runs on CPU, no network
calls once the model is cached). If the model can't be loaded (e.g. no
internet on first run, or the optional dependency isn't installed) we fall
back to a deterministic local hashing-vectorizer embedding so the rest of
the pipeline keeps working offline.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class _HashingFallbackEmbedder:
    """Fixed-dimension, dependency-light local embedder used as a fallback.

    Not as semantically rich as a transformer embedding, but fully local,
    deterministic, and keeps a stable vector dimension so it's a drop-in
    substitute inside the vector store.
    """

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm="l2"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype("float32")


class LocalEmbedder:
    """Wraps a local embedding model with a uniform `.encode()` interface."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._backend = "sentence-transformers"
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        except Exception as exc:  # noqa: BLE001 - broad on purpose for offline fallback
            logger.warning(
                "Falling back to local hashing embedder (sentence-transformers "
                "unavailable: %s). Install/cache '%s' for higher retrieval quality.",
                exc,
                self.model_name,
            )
            self._backend = "hashing-fallback"
            self._model = _HashingFallbackEmbedder(dim=settings.embedding_dim_fallback)
            self.dim = self._model.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if self._backend == "sentence-transformers":
            return np.asarray(
                self._model.encode(texts, normalize_embeddings=True), dtype="float32"
            )
        return self._model.encode(texts)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()


_singleton: Optional[LocalEmbedder] = None


def get_embedder() -> LocalEmbedder:
    global _singleton
    if _singleton is None:
        _singleton = LocalEmbedder()
    return _singleton
