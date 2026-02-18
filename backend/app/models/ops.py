from datetime import datetime
from typing import Literal

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


class CliSandboxResponse(BaseModel):
    status: str
    record: dict


CliCommandDecision = Literal["allow_once", "allow_always", "reject"]


class CliCommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=128)
    args: list[str] = Field(default_factory=list, max_length=20)
    timeout_seconds: int = Field(default=10, ge=1, le=120)


class CliCommandRequestResponse(BaseModel):
    status: Literal["completed", "approval_required"]
    approval_id: str | None = None
    record: dict | None = None


class CliCommandDecisionRequest(BaseModel):
    decision: CliCommandDecision


class CliCommandUnrestrictedModeRequest(BaseModel):
    enabled: bool
    acknowledged_danger: bool = False
    confirmation_text: str = ""


class CliCommandUnrestrictedModeResponse(BaseModel):
    enabled: bool
    updated_at: datetime
