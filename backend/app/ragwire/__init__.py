"""HealthAgentX's integration layer for the RAGWire library
(https://github.com/laxmimerit/ragwire), which provides the RAG stack
end-to-end: document loading/chunking, local embeddings, the local Qdrant
vector store, and local cross-encoder reranking - configured in
ragwire_config.yaml. This package just builds the shared pipeline and
exposes the interface HealthAgentX's Medical RAG Agent uses.
"""
from app.ragwire.retriever import RagWire, get_pipeline

__all__ = ["RagWire", "get_pipeline"]
