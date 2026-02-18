from datetime import datetime

from pydantic import BaseModel, Field


class BackupRequest(BaseModel):
    include_postgres: bool = True
    include_qdrant: bool = True


class BackupResponse(BaseModel):
    backup_dir: str
    created_at: datetime
    files: list[str]


class SandboxExecRequest(BaseModel):
    command: str = Field(min_length=1, max_length=128)
    args: list[str] = Field(default_factory=list, max_length=20)
    timeout_seconds: int = Field(default=10, ge=1, le=120)


class SandboxExecResponse(BaseModel):
    status: str
    record: dict


class InitResponse(BaseModel):
    status: str
    steps: dict[str, bool]
