from datetime import datetime

from pydantic import BaseModel


class KnowledgeStatusResponse(BaseModel):
    drop_folder: str
    files_detected: int
    last_scan_at: datetime | None
    qdrant_reachable: bool


class KnowledgeIngestRequest(BaseModel):
    full_rescan: bool = False
    limit: int = 100


class KnowledgeIngestResponse(BaseModel):
    accepted_files: int
    skipped_files: int
    started_at: datetime

