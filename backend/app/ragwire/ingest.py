"""Ingests the local medical knowledge base into the RAGWire vector store."""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.ragwire.retriever import get_pipeline

logger = logging.getLogger(__name__)

# Documents uploaded from the frontend are saved here, inside the knowledge
# base directory, so a later full re-ingest (ingest_knowledge_base) picks
# them up too. Kept separate from the curated built-in files for clarity.
UPLOADS_SUBDIR = "uploads"


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


def ingest_uploaded_files(files: list[tuple[str, bytes]]) -> dict:
    """Adds documents uploaded from the frontend to the knowledge base.

    Unlike ingest_knowledge_base(), this ingests only the newly uploaded
    files (via RAGWire's own ingest_documents()), so it's fast and doesn't
    touch anything already indexed - no pre-existing index or prior CLI step
    is required, since the pipeline auto-creates its collection on first use.

    Args:
        files: (filename, content) pairs, as read from the upload requests.

    Returns:
        RAGWire's IngestStats dict (processed, skipped, failed,
        chunks_created, errors, ...).

    Raises:
        ValueError: If a filename has an unsupported extension, or none of
            the given files could be saved.
    """
    pipeline = get_pipeline()
    allowed_extensions = set(pipeline.loader_extensions)

    upload_dir = Path(settings.knowledge_base_dir) / UPLOADS_SUBDIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for filename, content in files:
        # Strip any directory components a client might send, so an upload
        # can never write outside upload_dir.
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in allowed_extensions:
            raise ValueError(
                f"Unsupported file type '{suffix}' for '{safe_name}'. "
                f"Allowed: {', '.join(sorted(allowed_extensions))}"
            )

        destination = upload_dir / safe_name
        destination.write_bytes(content)
        saved_paths.append(str(destination))

    stats = pipeline.ingest_documents(saved_paths)
    logger.info("Uploaded-document ingestion stats: %s", stats)
    return stats
