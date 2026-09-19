from app.ragwire.document_loader import chunk_text
from app.ragwire.ingest import ingest_knowledge_base
from app.ragwire.retriever import RagWire


def test_chunk_text_basic():
    text = "a " * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_ingest_and_retrieve():
    count = ingest_knowledge_base(reset=True)
    assert count > 0

    ragwire = RagWire()
    assert ragwire.is_ready()

    results = ragwire.retrieve("chest pain shortness of breath emergency")
    assert len(results) > 0
    for hit in results:
        assert "content" in hit
        assert "source" in hit
        assert "score" in hit
