import asyncio
from datetime import datetime, timezone

from app.models.agent import ResearchResponse, SupervisorState, SwarmState, WorkerResult, WorkerState
from app.services.search_service import SearchService


class AgentSwarmService:
    def __init__(self, search_service: SearchService):
        self._search = search_service

    def _initial_state(self, query: str, search_mode: str | None) -> SwarmState:
        return SwarmState(
            query=query,
            search_mode=search_mode,
            supervisor=SupervisorState(status="running"),
            workers=[
                WorkerState(worker_id="worker-1", query=f"{query} overview"),
                WorkerState(worker_id="worker-2", query=f"{query} recent updates"),
            ],
            started_at=datetime.now(timezone.utc),
        )

    async def run_research(self, query: str, search_mode: str | None = None) -> ResearchResponse:
        state = self._initial_state(query, search_mode)
        worker_results = await asyncio.gather(
            self._run_worker(state.workers[0], state.search_mode),
            self._run_worker(state.workers[1], state.search_mode),
        )
        state.finished_at = datetime.now(timezone.utc)
        combined_summaries = " ".join(worker.summary or "" for worker in worker_results).strip()
        supervisor_summary = combined_summaries or "No findings."
        state.supervisor.status = "completed"
        state.supervisor.summary = supervisor_summary
        return ResearchResponse(
            query=state.query,
            supervisor_summary=supervisor_summary,
            workers=[
                WorkerResult(
                    worker_id=worker.worker_id,
                    summary=worker.summary or "No sources returned",
                    sources=worker.sources,
                )
                for worker in worker_results
            ],
        )

    async def _run_worker(self, worker: WorkerState, search_mode: str | None) -> WorkerState:
        worker.status = "running"
        worker.started_at = datetime.now(timezone.utc)
        try:
            response = await self._search.search(query=worker.query, mode=search_mode, limit=5)
            top = response.results[:3]
            worker.summary = " | ".join(result.title for result in top) if top else "No sources returned"
            worker.sources = [result.url for result in top]
            worker.status = "completed"
        except Exception as exc:  # noqa: BLE001
            worker.status = "failed"
            worker.error = str(exc)
            worker.summary = "Worker failed to retrieve sources"
            worker.sources = []
        finally:
            worker.finished_at = datetime.now(timezone.utc)
        return worker
