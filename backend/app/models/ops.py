from datetime import datetime

from pydantic import BaseModel, Field

from app.models.dependencies import DependencyItem


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


class OpsStatusConfigResponse(BaseModel):
    api_docs_enabled: bool
    require_auth: bool
    search_mode_default: str
    sensitive_actions_enabled: bool
    enable_mcp_services: bool
    enable_vllm: bool
    brave_api_key_configured: bool
    telegram_configured: bool
    google_oauth_configured: bool
    imap_configured: bool


class OpsStatusVersionResponse(BaseModel):
    app_version: str
    python_version: str
    environment: str


class OpsStatusResponse(BaseModel):
    status: str
    init_steps: dict[str, bool]
    dependencies: list[DependencyItem]
    config: OpsStatusConfigResponse
    version: OpsStatusVersionResponse
