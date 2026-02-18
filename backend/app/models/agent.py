from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    search_mode: str | None = None


class WorkerResult(BaseModel):
    worker_id: str
    summary: str
    sources: list[str]


class ResearchResponse(BaseModel):
    query: str
    supervisor_summary: str
    workers: list[WorkerResult]

