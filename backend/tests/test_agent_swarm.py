import asyncio

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


class SlowSearchService:
    async def search(self, query: str, mode: str | None = None, limit: int = 10) -> SearchResponse:  # noqa: ARG002
        _ = (query, mode, limit)
        await asyncio.sleep(0.05)
        return SearchResponse(mode="searxng_only", providers_used=["searxng"], results=[])


@pytest.mark.anyio
async def test_swarm_returns_two_workers_and_summary() -> None:
    service = AgentSwarmService(FakeSearchService())
    response = await service.run_research("docker backups", "searxng_only")
    assert response.query == "docker backups"
    assert len(response.workers) == 2
    assert "docker backups overview A" in response.supervisor_summary
    assert any(step.step == "supervisor_synthesis" for step in response.trace)


@pytest.mark.anyio
async def test_swarm_handles_worker_failures() -> None:
    service = AgentSwarmService(FailingSearchService())
    response = await service.run_research("docker backups", "searxng_only")
    assert len(response.workers) == 2
    assert all(worker.summary == "Worker failed to retrieve sources" for worker in response.workers)


@pytest.mark.anyio
async def test_swarm_respects_tool_call_budget() -> None:
    service = AgentSwarmService(FakeSearchService(), max_tool_calls=1)
    response = await service.run_research("docker backups", "searxng_only")
    assert len(response.workers) == 1
    assert any(step.step == "guardrail_tool_call_budget" for step in response.trace)


@pytest.mark.anyio
async def test_swarm_worker_timeout_guardrail() -> None:
    service = AgentSwarmService(SlowSearchService(), worker_timeout_seconds=0.01, overall_timeout_seconds=3)
    response = await service.run_research("docker backups", "searxng_only")
    assert len(response.workers) == 2
    assert all(worker.summary == "Worker timed out while retrieving sources" for worker in response.workers)


@pytest.mark.anyio
async def test_swarm_overall_timeout_guardrail() -> None:
    service = AgentSwarmService(SlowSearchService(), worker_timeout_seconds=2, overall_timeout_seconds=0.01)
    with pytest.raises(ValueError, match="Swarm execution timed out"):
        await service.run_research("docker backups", "searxng_only")


@pytest.mark.anyio
async def test_swarm_recursion_depth_guardrail() -> None:
    service = AgentSwarmService(FakeSearchService(), max_recursion_depth=0)
    with pytest.raises(ValueError, match="Recursion depth limit exceeded"):
        await service.run_research("docker backups", "searxng_only", recursion_depth=1)
