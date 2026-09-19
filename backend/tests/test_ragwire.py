from app.ragwire.retriever import RagWire


def test_ingest_and_retrieve(seeded_knowledge_base):
    ragwire = RagWire()
    assert ragwire.is_ready()

    results = ragwire.retrieve("chest pain shortness of breath emergency")
    assert len(results) > 0
    for hit in results:
        assert "content" in hit
        assert "source" in hit
        assert "score" in hit
