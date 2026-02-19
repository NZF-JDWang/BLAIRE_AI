import os

import pytest

from app.core.config import Settings
from app.models.search import SearchResult
from app.services.search_service import SearchError, SearchService


def _settings() -> Settings:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["INFERENCE_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["SEARCH_MODE_DEFAULT"] = "searxng_only"
    return Settings()


@pytest.mark.anyio
async def test_auto_fallback_uses_brave_when_searx_fails(monkeypatch) -> None:
    service = SearchService(_settings())

    async def fail_searx(query: str, limit: int):  # noqa: ANN001, ANN202
        raise SearchError("searx down")

    async def ok_brave(query: str, limit: int):  # noqa: ANN001, ANN202
        return [SearchResult(title="t", url="https://a", snippet="s", provider="brave")]

    monkeypatch.setattr(service, "_search_searxng", fail_searx)
    monkeypatch.setattr(service, "_search_brave", ok_brave)

    response = await service.search("test", mode="auto_fallback", limit=5)
    assert response.providers_used == ["brave"]
    assert len(response.results) == 1


@pytest.mark.anyio
async def test_parallel_merges_and_dedupes(monkeypatch) -> None:
    service = SearchService(_settings())

    async def searx(query: str, limit: int):  # noqa: ANN001, ANN202
        return [
            SearchResult(title="a", url="https://same", snippet="1", provider="searxng"),
            SearchResult(title="b", url="https://b", snippet="2", provider="searxng"),
        ]

    async def brave(query: str, limit: int):  # noqa: ANN001, ANN202
        return [SearchResult(title="c", url="https://same", snippet="3", provider="brave")]

    monkeypatch.setattr(service, "_search_searxng", searx)
    monkeypatch.setattr(service, "_search_brave", brave)

    response = await service.search("test", mode="parallel", limit=5)
    assert set(response.providers_used) == {"searxng", "brave"}
    assert len(response.results) == 2


