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

    # Local embedding model (used by RAGWire)
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Local cross-encoder reranker (used by RAGWire)
    reranker_model: str = os.environ.get(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # RAGWire vector store. A plain path runs Qdrant embedded on local disk;
    # an http(s):// URL points at a real Qdrant server instead.
    qdrant_url: str = os.environ.get("QDRANT_URL", str(BASE_DIR / "qdrant_local"))
    qdrant_collection: str = os.environ.get("QDRANT_COLLECTION", "medical_knowledge")

    # Path to the RAGWire library's own YAML config (see ragwire_config.yaml).
    ragwire_config_path: str = os.environ.get(
        "RAGWIRE_CONFIG_PATH", str(BASE_DIR / "ragwire_config.yaml")
    )
    knowledge_base_dir: str = os.environ.get(
        "KNOWLEDGE_BASE_DIR", str(BASE_DIR / "data" / "knowledge_base")
    )

    # App database
    database_url: str = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'health_agent_x.db'}"
    )

    # Loop Engineering
    max_verification_loops: int = _get_int("MAX_VERIFICATION_LOOPS", 3)
    # Final number of (reranked) chunks RAGWire returns per retrieve() call.
    # Candidate pool size before reranking is configured in ragwire_config.yaml
    # (retriever.rerank.fetch_k).
    retrieval_top_k: int = _get_int("RETRIEVAL_TOP_K", 4)

    # OCR
    tesseract_cmd: str = os.environ.get("TESSERACT_CMD", "")


settings = Settings()

# ragwire_config.yaml resolves ${VAR} placeholders straight from os.environ,
# so the values above (env var if set, our own default otherwise) need to be
# written back for RAGWire's own Config loader to see them - just reading
# os.environ.get() into a Python attribute above does not set the variable.
os.environ.setdefault("OLLAMA_HOST", settings.ollama_host)
os.environ.setdefault("OLLAMA_MODEL", settings.ollama_model)
os.environ.setdefault("EMBEDDING_MODEL", settings.embedding_model)
os.environ.setdefault("RERANKER_MODEL", settings.reranker_model)
os.environ.setdefault("QDRANT_URL", settings.qdrant_url)
os.environ.setdefault("QDRANT_COLLECTION", settings.qdrant_collection)
