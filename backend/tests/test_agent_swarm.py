import pytest

from app.models.search import SearchResponse, SearchResult
from app.services.agent_swarm import AgentSwarmService


class FakeSearchService:
    async def search(self, query: str, mode: str | None = None, limit: int = 10) -> SearchResponse:  # noqa: ARG002
        return SearchResponse(
            mode="searxng_only",
            providers_used=["searxng"],
            results=[
                SearchResult(title=f"{query} A", url="https://a", snippet="x", provider="searxng"),
                SearchResult(title=f"{query} B", url="https://b", snippet="y", provider="searxng"),
            ],
        )


class FailingSearchService:
    async def search(self, query: str, mode: str | None = None, limit: int = 10) -> SearchResponse:  # noqa: ARG002
        raise RuntimeError("search failed")


@pytest.mark.anyio
async def test_swarm_returns_two_workers_and_summary() -> None:
    service = AgentSwarmService(FakeSearchService())
    response = await service.run_research("docker backups", "searxng_only")
    assert response.query == "docker backups"
    assert len(response.workers) == 2
    assert "docker backups overview A" in response.supervisor_summary


@pytest.mark.anyio
async def test_swarm_handles_worker_failures() -> None:
    service = AgentSwarmService(FailingSearchService())
    response = await service.run_research("docker backups", "searxng_only")
    assert len(response.workers) == 2
    assert all(worker.summary == "Worker failed to retrieve sources" for worker in response.workers)
