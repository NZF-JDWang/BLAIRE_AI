from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.ops import BackupRequest, BackupResponse, SandboxExecRequest, SandboxExecResponse
from app.services.backup_service import BackupService
from app.services.sandbox_runner import LocalSandboxRunner, SandboxRunnerError

router = APIRouter(tags=["ops"])


@router.post("/ops/backup", response_model=BackupResponse)
async def run_backup(
    request: BackupRequest,
    _principal: Principal = Depends(require_roles("admin")),
) -> BackupResponse:
    settings = get_settings()
    service = BackupService(
        backup_root=settings.backup_path,
        database_url=settings.database_url.get_secret_value(),
        qdrant_url=settings.qdrant_url,
    )
    return service.run_backup(
        include_postgres=request.include_postgres,
        include_qdrant=request.include_qdrant,
    )


@router.post("/ops/sandbox/execute", response_model=SandboxExecResponse)
async def sandbox_execute(
    request: SandboxExecRequest,
    _principal: Principal = Depends(require_roles("admin")),
) -> SandboxExecResponse:
    settings = get_settings()
    runner = LocalSandboxRunner(settings.sandbox_allowed_commands_list())
    try:
        record = await runner.run(
            command=request.command,
            args=request.args,
            timeout_seconds=request.timeout_seconds,
        )
    except SandboxRunnerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SandboxExecResponse(status="completed", record=record.to_dict())
