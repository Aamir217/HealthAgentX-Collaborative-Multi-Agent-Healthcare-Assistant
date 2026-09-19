"""RagWire: the retrieve-and-rerank facade used by the Medical RAG Agent.

Pipeline: embed query (local) -> ANN search (local Chroma) -> local
cross-encoder rerank -> top-N evidence chunks with provenance.
"""
from __future__ import annotations

from app.config import settings
from app.ragwire.embeddings import get_embedder
from app.ragwire.reranker import get_reranker
from app.ragwire.vector_store import get_vector_store


class RagWire:
    def __init__(self):
        self.embedder = get_embedder()
        self.reranker = get_reranker()
        self.store = get_vector_store()

    def retrieve(self, query: str, top_k: int | None = None, rerank_top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.retrieval_top_k
        rerank_top_k = rerank_top_k or settings.rerank_top_k

        query_embedding = self.embedder.encode_one(query)
        candidates = self.store.query(query_embedding, top_k=top_k)
        if not candidates:
            return []

        passages = [c["content"] for c in candidates]
        reranked = self.reranker.rerank(query, passages, top_k=rerank_top_k)

        results = []
        for idx, score in reranked:
            candidate = candidates[idx]
            results.append(
                {
                    "content": candidate["content"],
                    "source": candidate["source"],
                    "score": round(float(score), 4),
                    "query": query,
                }
            )
        return results

    def is_ready(self) -> bool:
        return self.store.count() > 0
