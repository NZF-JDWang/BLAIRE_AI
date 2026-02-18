from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class SwarmRunRecord:
    run_id: str
    query: str
    created_at: str
    supervisor_summary: str
    workers: list[dict[str, Any]]
    trace: list[dict[str, Any]]


class SwarmTraceStore:
    def __init__(self, max_runs: int = 50):
        self._runs: deque[SwarmRunRecord] = deque(maxlen=max_runs)
        self._lock = Lock()

    def add_run(self, *, query: str, supervisor_summary: str, workers: list[dict[str, Any]], trace: list[dict[str, Any]]) -> str:
        run_id = str(uuid4())
        record = SwarmRunRecord(
            run_id=run_id,
            query=query,
            created_at=datetime.now(timezone.utc).isoformat(),
            supervisor_summary=supervisor_summary,
            workers=workers,
            trace=trace,
        )
        with self._lock:
            self._runs.appendleft(record)
        return run_id

    def list_recent(self, limit: int = 20) -> list[SwarmRunRecord]:
        with self._lock:
            return list(self._runs)[:limit]

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


swarm_trace_store = SwarmTraceStore()
