import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

_tmp_dir = tempfile.mkdtemp(prefix="healthagentx_test_")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(_tmp_dir) / "chroma_db"))
os.environ.setdefault("CHROMA_COLLECTION", "test_medical_knowledge")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{Path(_tmp_dir) / 'test.db'}")
os.environ.setdefault("KNOWLEDGE_BASE_DIR", str(BACKEND_DIR.parent / "data" / "knowledge_base"))
# Force the offline-friendly fallback backends so tests run without network
# access to download sentence-transformers/cross-encoder models, and without
# a local Ollama server.
os.environ.setdefault("OLLAMA_HOST", "http://localhost:1")


@pytest.fixture(scope="session", autouse=True)
def _seed_knowledge_base():
    from app.ragwire.ingest import ingest_knowledge_base

    ingest_knowledge_base(reset=True)
