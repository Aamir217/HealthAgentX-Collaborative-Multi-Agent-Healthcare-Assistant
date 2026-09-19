"""Thin wrapper around the RAGWire library (https://github.com/laxmimerit/ragwire).

RAGWire owns the entire RAG stack end to end - document loading/chunking,
local embeddings, the local Qdrant vector store, and local cross-encoder
reranking - all configured declaratively in ragwire_config.yaml. This module
just builds one shared pipeline instance and exposes the narrow interface
the Medical RAG Agent needs (`retrieve`), translating RAGWire's LangChain
`Document` results into the plain dicts the rest of the app expects.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_pipeline = None


def get_pipeline():
    """Builds (once) and returns the shared RAGWire pipeline instance.

    Constructing it loads the embedding model and initializes the local
    Qdrant store, so it's expensive - hence the module-level singleton.
    """
    global _pipeline
    if _pipeline is None:
        from ragwire import RAGWire

        logger.info("Initializing RAGWire pipeline from %s", settings.ragwire_config_path)
        _pipeline = RAGWire(settings.ragwire_config_path)
    return _pipeline


class RagWire:
    """Facade used by the Medical RAG Agent: `RagWire().retrieve(query)`."""

    def __init__(self):
        self.pipeline = get_pipeline()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        top_k = top_k or settings.retrieval_top_k
        documents = self.pipeline.retrieve(query, top_k=top_k)

        results = []
        for doc in documents:
            metadata = doc.metadata or {}
            results.append(
                {
                    "content": doc.page_content,
                    "source": metadata.get("file_name") or metadata.get("source", "unknown"),
                    # Present only when a reranker is configured (it is, by default).
                    "score": round(float(metadata.get("rerank_score", 0.0)), 4),
                    "query": query,
                }
            )
        return results

    def is_ready(self) -> bool:
        try:
            return self.pipeline.get_stats().get("total_documents", 0) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read RAGWire collection stats: %s", exc)
            return False
