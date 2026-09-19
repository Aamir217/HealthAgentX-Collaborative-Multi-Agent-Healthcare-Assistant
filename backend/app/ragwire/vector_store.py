"""Persistent local vector store backed by ChromaDB.

Embeddings are computed ourselves via `LocalEmbedder` and passed in explicitly
(rather than relying on Chroma's embedding_function) so the fallback logic in
embeddings.py is always the single source of truth for vectorization.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import chromadb

from app.config import settings


def _make_id(source: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha1(f"{source}:{chunk_index}:{content[:64]}".encode()).hexdigest()
    return digest


class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """chunks: [{"content": str, "source": str, "chunk_index": int, **meta}]"""
        if not chunks:
            return 0
        ids = [_make_id(c["source"], c["chunk_index"], c["content"]) for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        return len(ids)

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self._collection.count() == 0:
            return []
        top_k = min(top_k, self._collection.count())
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"content": doc, "source": meta.get("source", "unknown"), "distance": dist})
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


_singleton: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _singleton
    if _singleton is None:
        _singleton = VectorStore()
    return _singleton
