"""Local cross-encoder reranker with a lexical-overlap fallback."""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _WORD_RE.findall(text)}


class LocalReranker:
    """Reranks (query, passage) pairs. Falls back to Jaccard token overlap
    when the local cross-encoder model cannot be loaded (e.g. offline)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.reranker_model
        self._backend = "cross-encoder"
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falling back to lexical-overlap reranker (cross-encoder "
                "unavailable: %s).",
                exc,
            )
            self._backend = "lexical-fallback"
            self._model = None

    def rerank(self, query: str, passages: list[str], top_k: int) -> list[tuple[int, float]]:
        """Returns list of (original_index, score) sorted best-first."""
        if not passages:
            return []
        if self._backend == "cross-encoder":
            pairs = [(query, p) for p in passages]
            scores = self._model.predict(pairs)
            scored = list(enumerate(float(s) for s in scores))
        else:
            q_tokens = _tokenize(query)
            scored = []
            for i, passage in enumerate(passages):
                p_tokens = _tokenize(passage)
                if not q_tokens or not p_tokens:
                    scored.append((i, 0.0))
                    continue
                overlap = len(q_tokens & p_tokens) / len(q_tokens | p_tokens)
                scored.append((i, overlap))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


_singleton: Optional[LocalReranker] = None


def get_reranker() -> LocalReranker:
    global _singleton
    if _singleton is None:
        _singleton = LocalReranker()
    return _singleton
