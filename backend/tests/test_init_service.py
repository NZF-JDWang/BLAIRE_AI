import os

import pytest

from app.core.config import Settings
from app.services.init_service import InitService


def _settings() -> Settings:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["INFERENCE_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    return Settings()


@pytest.mark.anyio
async def test_init_service_runs_all_steps(monkeypatch) -> None:
    calls: list[str] = []

    async def approval(self):  # noqa: ANN001, ANN202
        _ = self
        calls.append("approval")

    async def prefs(self):  # noqa: ANN001, ANN202
        _ = self
        calls.append("preferences")

    async def meta(self):  # noqa: ANN001, ANN202
        _ = self
        calls.append("metadata")

    async def qdrant(self):  # noqa: ANN001, ANN202
        _ = self
        calls.append("qdrant")

    monkeypatch.setattr("app.services.init_service.ApprovalService.init_schema", approval)
    monkeypatch.setattr("app.services.init_service.PreferencesService.init_schema", prefs)
    monkeypatch.setattr("app.services.init_service.MetadataStoreService.init_schema", meta)
    monkeypatch.setattr("app.services.init_service.QdrantBootstrapService.ensure_collection", qdrant)

    result = await InitService(_settings()).run()
    assert result["approval_schema_ready"] is True
    assert result["preferences_schema_ready"] is True
    assert result["metadata_schema_ready"] is True
    assert result["qdrant_collection_ready"] is True
    assert calls == ["approval", "preferences", "metadata", "qdrant"]

