from datetime import datetime

from pydantic import BaseModel


class BackupRequest(BaseModel):
    include_postgres: bool = True
    include_qdrant: bool = True


class BackupResponse(BaseModel):
    backup_dir: str
    created_at: datetime
    files: list[str]
