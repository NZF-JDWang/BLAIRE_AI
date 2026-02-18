from pathlib import Path

import pytest

from app.rag.retrieval import IngestionPipeline, RetrievalService


class FakeOllama:
    async def embed(self, model: str, text: str):  # noqa: ANN001, ANN202
        _ = model
        return [float(len(text)), 1.0, 2.0]


class FakeVectorStore:
    def __init__(self):
        self.upserted = 0

    async def upsert_chunks(self, path: Path, chunks, embeddings):  # noqa: ANN001, ANN202
        _ = path
        self.upserted += len(chunks)
        assert len(chunks) == len(embeddings)
        return len(chunks)

    async def search(self, vector, limit=5):  # noqa: ANN001, ANN202
        _ = vector
        return [
            {
                "score": 0.9,
                "source_path": "/tmp/a.md",
                "source_name": "a.md",
                "file_type": ".md",
                "chunk_index": 0,
                "text": "chunk text",
                "last_modified": "2026-01-01T00:00:00+00:00",
            }
        ][:limit]


@pytest.mark.anyio
async def test_ingestion_pipeline_indexes_chunks(tmp_path: Path) -> None:
    doc = tmp_path / "a.md"
    doc.write_text("hello " * 300, encoding="utf-8")
    vector = FakeVectorStore()
    pipeline = IngestionPipeline(ollama_client=FakeOllama(), vector_store=vector, embedding_model="embed")
    indexed = await pipeline.ingest_file(doc)
    assert indexed > 0
    assert vector.upserted == indexed


@pytest.mark.anyio
async def test_retrieval_service_returns_results() -> None:
    service = RetrievalService(ollama_client=FakeOllama(), vector_store=FakeVectorStore(), embedding_model="embed")
    rows = await service.retrieve("what is this", limit=3)
    assert len(rows) == 1
    assert rows[0].source_name == "a.md"

