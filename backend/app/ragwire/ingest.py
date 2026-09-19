"""Ingests the local medical knowledge base into the RAGWire vector store."""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.ragwire.document_loader import chunk_text, extract_text
from app.ragwire.embeddings import get_embedder
from app.ragwire.vector_store import get_vector_store

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def ingest_knowledge_base(directory: str | None = None, reset: bool = False) -> int:
    directory_path = Path(directory or settings.knowledge_base_dir)
    if not directory_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory_path}")

    store = get_vector_store()
    if reset:
        store.reset()

    embedder = get_embedder()
    total_chunks = 0

    for file_path in sorted(directory_path.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES or not file_path.is_file():
            continue
        content = file_path.read_bytes()
        text = extract_text(file_path.name, content)
        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = embedder.encode(chunks).tolist()
        chunk_dicts = [
            {"content": chunk, "source": file_path.name, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]
        added = store.add_chunks(chunk_dicts, embeddings)
        total_chunks += added
        logger.info("Ingested %s chunks from %s", added, file_path.name)

    return total_chunks
