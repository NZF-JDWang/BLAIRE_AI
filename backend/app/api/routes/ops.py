import platform

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.services.dependency_checks import collect_dependency_status
from app.models.ops import (
    BackupRequest,
    BackupResponse,
    CliSandboxResponse,
    InitResponse,
    OpsStatusConfigResponse,
    OpsStatusResponse,
    OpsStatusVersionResponse,
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


@router.get("/ops/status", response_model=OpsStatusResponse)
async def ops_status(
    _principal: Principal = Depends(require_roles("admin")),
) -> OpsStatusResponse:
    settings = get_settings()
    init_steps = await InitService(settings).run()
    deps = await collect_dependency_status(settings)

    config = OpsStatusConfigResponse(
        api_docs_enabled=settings.api_docs_enabled,
        require_auth=settings.require_auth,
        search_mode_default=settings.search_mode_default,
        sensitive_actions_enabled=settings.sensitive_actions_enabled,
        enable_mcp_services=settings.enable_mcp_services,
        enable_vllm=settings.enable_vllm,
        brave_api_key_configured=bool(settings.brave_api_key and settings.brave_api_key.get_secret_value()),
        telegram_configured=bool(settings.telegram_bot_token and settings.telegram_bot_token.get_secret_value()),
        google_oauth_configured=bool(settings.google_oauth_token and settings.google_oauth_token.get_secret_value()),
        imap_configured=bool(settings.imap_host and settings.imap_user and settings.imap_password),
    )
    version = OpsStatusVersionResponse(
        app_version="0.1.0",
        python_version=platform.python_version(),
        environment=settings.app_env,
    )
    required_deps_ok = all(item.ok for item in deps.dependencies if item.required and item.enabled)
    status = "ready" if all(init_steps.values()) and required_deps_ok else "degraded"
    return OpsStatusResponse(
        status=status,
        init_steps=init_steps,
        dependencies=deps.dependencies,
        config=config,
        version=version,
    )


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
