"""Central configuration for HealthAgentX, loaded from environment variables.

Kept as a single module (rather than a pydantic BaseSettings class) so it has
zero hard dependency on any optional package and can be imported early.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    # Local LLM (Ollama)
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    llm_timeout_s: int = _get_int("LLM_TIMEOUT_S", 60)

    # Local embedding model
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim_fallback: int = _get_int("EMBEDDING_DIM_FALLBACK", 384)

    # Local cross-encoder reranker
    reranker_model: str = os.environ.get(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # RAGWire vector store (Chroma)
    chroma_persist_dir: str = os.environ.get(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")
    )
    chroma_collection: str = os.environ.get("CHROMA_COLLECTION", "medical_knowledge")
    knowledge_base_dir: str = os.environ.get(
        "KNOWLEDGE_BASE_DIR", str(BASE_DIR / "data" / "knowledge_base")
    )

    # App database
    database_url: str = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'health_agent_x.db'}"
    )

    # Loop Engineering
    max_verification_loops: int = _get_int("MAX_VERIFICATION_LOOPS", 3)
    retrieval_top_k: int = _get_int("RETRIEVAL_TOP_K", 8)
    rerank_top_k: int = _get_int("RERANK_TOP_K", 4)

    # OCR
    tesseract_cmd: str = os.environ.get("TESSERACT_CMD", "")


settings = Settings()
