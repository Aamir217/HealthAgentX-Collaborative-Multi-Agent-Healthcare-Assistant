"""Ingests the local medical knowledge base into the RAGWire vector store."""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.ragwire.retriever import get_pipeline

logger = logging.getLogger(__name__)


def ingest_knowledge_base(directory: str | None = None, reset: bool = False) -> int:
    directory_path = Path(directory or settings.knowledge_base_dir)
    if not directory_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory_path}")

    pipeline = get_pipeline()

    if reset:
        # Recreate the collection through the already-open pipeline/client
        # rather than constructing a second RAGWire instance: local Qdrant
        # storage allows exactly one reader/writer process, so opening a
        # second client against the same path would deadlock.
        store = pipeline.vectorstore_wrapper
        if store.collection_exists():
            store.delete_collection()
            logger.info("Deleted existing collection '%s' for reset", store.collection_name)
        use_sparse = pipeline.config.get("vectorstore", {}).get("use_sparse", False)
        store.create_collection(use_sparse=use_sparse)
        pipeline.vectorstore = store.get_store(use_sparse=use_sparse)

    stats = pipeline.ingest_directory(str(directory_path), recursive=True)
    logger.info("Knowledge base ingestion stats: %s", stats)
    return stats["chunks_created"]
