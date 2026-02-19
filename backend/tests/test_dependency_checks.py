import os

import pytest

from app.core.config import Settings
from app.services import dependency_checks
from app.services.dependency_checks import collect_dependency_status


def _settings() -> Settings:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["INFERENCE_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["BRAVE_API_KEY"] = ""
    return Settings()


@pytest.mark.anyio
async def test_dependency_status_includes_brave_config(monkeypatch) -> None:
    async def fake_check_http(name: str, url: str, *, required: bool, enabled: bool, timeout: float = 3.0):  # noqa: ARG001, ANN001, ANN202
        from app.models.dependencies import DependencyItem

        return DependencyItem(name=name, ok=True, detail="reachable", required=required, enabled=enabled)

    monkeypatch.setattr(dependency_checks, "_check_http", fake_check_http)
    status = await collect_dependency_status(_settings())
    brave = [dep for dep in status.dependencies if dep.name == "brave_api_key"][0]
    assert brave.ok is True
    assert brave.detail == "disabled"
    assert brave.required is False
    assert brave.enabled is False


