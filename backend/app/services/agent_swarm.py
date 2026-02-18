import asyncio

from app.models.agent import ResearchResponse, WorkerResult
from app.services.search_service import SearchService


class AgentSwarmService:
    def __init__(self, search_service: SearchService):
        self._search = search_service

    async def run_research(self, query: str, search_mode: str | None = None) -> ResearchResponse:
        worker_queries = [
            f"{query} overview",
            f"{query} recent updates",
        ]
        worker_results = await asyncio.gather(
            self._run_worker("worker-1", worker_queries[0], search_mode),
            self._run_worker("worker-2", worker_queries[1], search_mode),
        )

        combined_summaries = " ".join(result.summary for result in worker_results).strip()
        supervisor_summary = combined_summaries or "No findings."
        return ResearchResponse(
            query=query,
            supervisor_summary=supervisor_summary,
            workers=worker_results,
        )

    async def _run_worker(self, worker_id: str, query: str, search_mode: str | None) -> WorkerResult:
        response = await self._search.search(query=query, mode=search_mode, limit=5)
        top = response.results[:3]
        summary = " | ".join(result.title for result in top) if top else "No sources returned"
        sources = [result.url for result in top]
        return WorkerResult(worker_id=worker_id, summary=summary, sources=sources)

