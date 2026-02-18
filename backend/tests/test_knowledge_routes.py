import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.rag.ingestion import IngestionResult, WatchIngestionResult
from app.rag.retrieval import RetrievalItem


def _set_required_env(drop_folder: Path) -> None:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["API_ALLOWED_HOSTS"] = "testserver,localhost,127.0.0.1,backend"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["REQUIRE_AUTH"] = "true"
    os.environ["USER_API_KEYS"] = "test-user-key"
    os.environ["ADMIN_API_KEYS"] = "test-admin-key"
    os.environ["DROP_FOLDER"] = str(drop_folder)


def test_knowledge_retrieve_route(monkeypatch, tmp_path: Path) -> None:
    _set_required_env(tmp_path)
    from app.core.config import get_settings
    from app.main import create_app
    from app.rag.retrieval import RetrievalService

    get_settings.cache_clear()

    async def fake_retrieve(self, query: str, limit: int = 5):  # noqa: ANN001, ANN202
        _ = (self, query, limit)
        return [
            RetrievalItem(
                source_path=str(tmp_path / "note.md"),
                source_name="note.md",
                file_type=".md",
                chunk_index=0,
                score=0.87,
                text="retrieved chunk",
                last_modified="2026-02-18T00:00:00+00:00",
            )
        ]

    monkeypatch.setattr(RetrievalService, "retrieve", fake_retrieve)

    client = TestClient(create_app())
    response = client.post(
        "/knowledge/retrieve",
        headers={"X-API-Key": "test-user-key"},
        json={"query": "test question", "limit": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "test question"
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["source_name"] == "note.md"


def test_knowledge_ingest_route(monkeypatch, tmp_path: Path) -> None:
    _set_required_env(tmp_path)
    from app.core.config import get_settings
    from app.main import create_app
    from app.rag.ingestion import DropFolderIngestionService

    get_settings.cache_clear()

    async def fake_ingest_with_pipeline(self, *, pipeline, full_rescan: bool = False, limit: int = 100):  # noqa: ANN001, ANN202
        _ = (self, pipeline, full_rescan, limit)
        return IngestionResult(
            accepted_files=2,
            skipped_files=1,
            started_at=datetime.now(timezone.utc),
            chunks_indexed=6,
        )

    monkeypatch.setattr(DropFolderIngestionService, "ingest_with_pipeline", fake_ingest_with_pipeline)

    client = TestClient(create_app())
    response = client.post(
        "/knowledge/ingest",
        headers={"X-API-Key": "test-user-key"},
        json={"full_rescan": True, "limit": 20},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted_files"] == 2
    assert payload["skipped_files"] == 1
    assert payload["chunks_indexed"] == 6


def test_knowledge_ingest_watcher_mode(monkeypatch, tmp_path: Path) -> None:
    _set_required_env(tmp_path)
    from app.core.config import get_settings
    from app.main import create_app
    from app.rag.ingestion import DropFolderIngestionService

    get_settings.cache_clear()

    async def fake_ingest_changed_with_retry(  # noqa: ANN001, ANN202
        self,
        *,
        pipeline,
        limit: int = 100,
        debounce_seconds: int = 10,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 300,
        current_time_ts: float | None = None,
    ):
        _ = (
            self,
            pipeline,
            limit,
            debounce_seconds,
            retry_base_seconds,
            retry_max_seconds,
            current_time_ts,
        )
        return WatchIngestionResult(
            scanned_files=3,
            indexed_files=1,
            skipped_files=2,
            failed_files=0,
            chunks_indexed=4,
            started_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(DropFolderIngestionService, "ingest_changed_with_retry", fake_ingest_changed_with_retry)

    client = TestClient(create_app())
    response = client.post(
        "/knowledge/ingest",
        headers={"X-API-Key": "test-user-key"},
        json={
            "use_watcher": True,
            "limit": 50,
            "debounce_seconds": 12,
            "retry_base_seconds": 6,
            "retry_max_seconds": 60,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted_files"] == 3
    assert payload["indexed_files"] == 1
    assert payload["skipped_files"] == 2
    assert payload["failed_files"] == 0
    assert payload["chunks_indexed"] == 4
