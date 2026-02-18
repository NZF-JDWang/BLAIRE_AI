from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    search_mode: str | None = None


class WorkerResult(BaseModel):
    worker_id: str
    summary: str
    sources: list[str]


class ConsolidatedCitation(BaseModel):
    url: str
    worker_ids: list[str]
    occurrences: int


class SwarmTraceStep(BaseModel):
    step: str
    status: Literal["started", "completed", "failed", "skipped"]
    timestamp: datetime
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    query: str
    supervisor_summary: str
    workers: list[WorkerResult]
    citations: list[ConsolidatedCitation] = Field(default_factory=list)
    trace: list[SwarmTraceStep] = Field(default_factory=list)


class SwarmLiveRun(BaseModel):
    run_id: str
    query: str
    created_at: str
    supervisor_summary: str
    workers: list[WorkerResult]
    trace: list[SwarmTraceStep]


class SwarmLiveResponse(BaseModel):
    runs: list[SwarmLiveRun]


WorkerStatus = Literal["pending", "running", "completed", "failed"]
SupervisorStatus = Literal["pending", "running", "completed", "failed"]


class WorkerState(BaseModel):
    worker_id: str
    query: str
    status: WorkerStatus = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    summary: str | None = None
    sources: list[str] = Field(default_factory=list)
    error: str | None = None


class SupervisorState(BaseModel):
    status: SupervisorStatus = "pending"
    summary: str | None = None


class SwarmState(BaseModel):
    query: str
    search_mode: str | None = None
    supervisor: SupervisorState
    workers: list[WorkerState]
    started_at: datetime
    finished_at: datetime | None = None
