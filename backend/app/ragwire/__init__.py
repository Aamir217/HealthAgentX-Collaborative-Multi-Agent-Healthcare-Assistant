"""RAGWire: HealthAgentX's local retrieval-augmented generation subsystem.

Wires together local embeddings, a persistent local vector store (ChromaDB),
a local cross-encoder reranker, and document ingestion (PDF/OCR) so that
sensitive patient data and medical knowledge never leave the machine.
"""
from app.ragwire.retriever import RagWire

__all__ = ["RagWire"]
