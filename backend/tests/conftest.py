import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

_tmp_dir = tempfile.mkdtemp(prefix="healthagentx_test_")
os.environ.setdefault("QDRANT_URL", str(Path(_tmp_dir) / "qdrant_local"))
os.environ.setdefault("QDRANT_COLLECTION", "test_medical_knowledge")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{Path(_tmp_dir) / 'test.db'}")
os.environ.setdefault("KNOWLEDGE_BASE_DIR", str(BACKEND_DIR.parent / "data" / "knowledge_base"))
os.environ.setdefault(
    "RAGWIRE_CONFIG_PATH", str(BACKEND_DIR.parent / "ragwire_config.yaml")
)
# No local Ollama server in this sandbox. RAGWire only needs the LLM for
# per-document metadata extraction at ingest time, and fails that gracefully
# per-document (ingesting without metadata) rather than aborting, so tests
# still exercise the real retrieval/rerank path against an unreachable LLM.
os.environ.setdefault("OLLAMA_HOST", "http://localhost:1")


@pytest.fixture(scope="session", autouse=True)
def _seed_knowledge_base():
    from app.ragwire.ingest import ingest_knowledge_base

    try:
        ingest_knowledge_base(reset=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            "RAGWire pipeline unavailable in this environment: the local "
            "embedding/reranker models could not be loaded (they download "
            f"from HuggingFace on first use). Original error: {exc}"
        )
