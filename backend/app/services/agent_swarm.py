import asyncio
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.models.agent import (
    ConsolidatedCitation,
    ResearchResponse,
    SupervisorState,
    SwarmState,
    SwarmTraceStep,
    WorkerResult,
    WorkerState,
)
from app.services.search_service import SearchService


class AgentSwarmService:
    def __init__(
        self,
        search_service: SearchService,
        *,
        max_tool_calls: int = 4,
        max_recursion_depth: int = 2,
        worker_timeout_seconds: int = 12,
        overall_timeout_seconds: int = 20,
    ):
        self._search = search_service
        self._max_tool_calls = max_tool_calls
        self._max_recursion_depth = max_recursion_depth
        self._worker_timeout_seconds = worker_timeout_seconds
        self._overall_timeout_seconds = overall_timeout_seconds
        self._logger = get_logger(component="agent_swarm")

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

    def _trace(
        self,
        trace_steps: list[SwarmTraceStep],
        *,
        step: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = details or {}
        normalized = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (str, int, float, bool))
        }
        event = SwarmTraceStep(
            step=step,
            status=status,  # type: ignore[arg-type]
            timestamp=datetime.now(timezone.utc),
            details=normalized,
        )
        trace_steps.append(event)
        self._logger.info(
            "swarm_trace_step",
            step=event.step,
            status=event.status,
            details=event.details,
        )

    async def run_research(
        self,
        query: str,
        search_mode: str | None = None,
        recursion_depth: int = 0,
    ) -> ResearchResponse:
        trace_steps: list[SwarmTraceStep] = []
        self._trace(
            trace_steps,
            step="swarm_start",
            status="started",
            details={"recursion_depth": recursion_depth, "search_mode": search_mode or "default"},
        )
        if recursion_depth > self._max_recursion_depth:
            self._trace(
                trace_steps,
                step="guardrail_recursion_depth",
                status="failed",
                details={"max_recursion_depth": self._max_recursion_depth},
            )
            raise ValueError("Recursion depth limit exceeded")

        state = self._initial_state(query, search_mode)
        self._trace(trace_steps, step="state_initialized", status="completed", details={"worker_count": len(state.workers)})
        effective_workers = state.workers[: self._max_tool_calls]
        skipped_workers = max(0, len(state.workers) - len(effective_workers))
        if skipped_workers:
            self._trace(
                trace_steps,
                step="guardrail_tool_call_budget",
                status="skipped",
                details={"max_tool_calls": self._max_tool_calls, "skipped_workers": skipped_workers},
            )

        async def _run_workers() -> list[WorkerState]:
            return await asyncio.gather(
                *[self._run_worker(worker, state.search_mode, trace_steps) for worker in effective_workers]
            )

        try:
            worker_results = await asyncio.wait_for(_run_workers(), timeout=self._overall_timeout_seconds)
        except asyncio.TimeoutError:
            self._trace(
                trace_steps,
                step="guardrail_overall_timeout",
                status="failed",
                details={"overall_timeout_seconds": self._overall_timeout_seconds},
            )
            raise ValueError("Swarm execution timed out")

        state.finished_at = datetime.now(timezone.utc)
        combined_summaries = " ".join(worker.summary or "" for worker in worker_results).strip()
        supervisor_summary = combined_summaries or "No findings."
        state.supervisor.status = "completed"
        state.supervisor.summary = supervisor_summary
        citations = self._merge_citations(worker_results)
        self._trace(
            trace_steps,
            step="supervisor_synthesis",
            status="completed",
            details={"worker_results": len(worker_results), "citation_count": len(citations)},
        )
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
            citations=citations,
            trace=trace_steps,
        )

    def _merge_citations(self, workers: list[WorkerState]) -> list[ConsolidatedCitation]:
        by_url: dict[str, set[str]] = {}
        for worker in workers:
            for source in worker.sources:
                by_url.setdefault(source, set()).add(worker.worker_id)
        return [
            ConsolidatedCitation(
                url=url,
                worker_ids=sorted(worker_ids),
                occurrences=len(worker_ids),
            )
            for url, worker_ids in sorted(by_url.items(), key=lambda item: item[0])
        ]

    async def _run_worker(
        self,
        worker: WorkerState,
        search_mode: str | None,
        trace_steps: list[SwarmTraceStep],
    ) -> WorkerState:
        worker.status = "running"
        worker.started_at = datetime.now(timezone.utc)
        self._trace(
            trace_steps,
            step=f"{worker.worker_id}_search",
            status="started",
            details={"query": worker.query},
        )
        try:
            response = await asyncio.wait_for(
                self._search.search(query=worker.query, mode=search_mode, limit=5),
                timeout=self._worker_timeout_seconds,
            )
            top = response.results[:3]
            worker.summary = " | ".join(result.title for result in top) if top else "No sources returned"
            worker.sources = [result.url for result in top]
            worker.status = "completed"
            self._trace(
                trace_steps,
                step=f"{worker.worker_id}_search",
                status="completed",
                details={"source_count": len(worker.sources)},
            )
        except asyncio.TimeoutError:
            worker.status = "failed"
            worker.error = "worker_timeout"
            worker.summary = "Worker timed out while retrieving sources"
            worker.sources = []
            self._trace(
                trace_steps,
                step=f"{worker.worker_id}_search",
                status="failed",
                details={"worker_timeout_seconds": self._worker_timeout_seconds},
            )
        except Exception as exc:  # noqa: BLE001
            worker.status = "failed"
            worker.error = str(exc)
            worker.summary = "Worker failed to retrieve sources"
            worker.sources = []
            self._trace(
                trace_steps,
                step=f"{worker.worker_id}_search",
                status="failed",
                details={"error": str(exc)},
            )
        finally:
            worker.finished_at = datetime.now(timezone.utc)
        return worker
