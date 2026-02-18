from datetime import datetime

from pydantic import BaseModel


class KnowledgeStatusResponse(BaseModel):
    drop_folder: str
    files_detected: int
    last_scan_at: datetime | None
    qdrant_reachable: bool
    obsidian_vault_path: str
    obsidian_files_detected: int
    obsidian_last_scan_at: datetime | None


class KnowledgeIngestRequest(BaseModel):
    full_rescan: bool = False
    limit: int = 100


class KnowledgeIngestResponse(BaseModel):
    accepted_files: int
    skipped_files: int
    started_at: datetime
    chunks_indexed: int = 0


class KnowledgeRetrieveRequest(BaseModel):
    query: str
    limit: int = 5


class KnowledgeCitation(BaseModel):
    source_path: str
    source_name: str
    file_type: str
    chunk_index: int
    score: float
    text: str
    last_modified: str


class KnowledgeRetrieveResponse(BaseModel):
    query: str
    citations: list[KnowledgeCitation]


class ObsidianReindexRequest(BaseModel):
    full_rescan: bool = False
    limit: int = 5000


class ObsidianReindexResponse(BaseModel):
    scanned_files: int
    indexed_files: int
    unchanged_files: int
    skipped_files: int
    failed_files: int
    chunks_indexed: int
    started_at: datetime
