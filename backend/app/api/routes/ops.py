from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.ops import (
    BackupRequest,
    BackupResponse,
    CliSandboxResponse,
    InitResponse,
    SandboxExecRequest,
    SandboxExecResponse,
)
from app.services.backup_service import BackupService
from app.services.cli_sandbox import CliSandboxError, CliSandboxRunner
from app.services.init_service import InitService
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


@router.post("/ops/init", response_model=InitResponse)
async def run_init(
    _principal: Principal = Depends(require_roles("admin")),
) -> InitResponse:
    settings = get_settings()
    steps = await InitService(settings).run()
    return InitResponse(status="completed", steps=steps)


@router.post("/ops/cli/execute", response_model=CliSandboxResponse)
async def cli_execute(
    request: SandboxExecRequest,
    _principal: Principal = Depends(require_roles("admin")),
) -> CliSandboxResponse:
    settings = get_settings()
    if not settings.cli_sandbox_enabled:
        raise HTTPException(status_code=503, detail="CLI sandbox is disabled")

    runner = CliSandboxRunner(
        backend=settings.cli_sandbox_backend,
        allowed_commands=settings.sandbox_allowed_commands_list(),
    )
    try:
        record = runner.run(command=request.command, args=request.args, timeout_seconds=request.timeout_seconds)
    except CliSandboxError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CliSandboxResponse(status="completed", record=record.to_dict())
