import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.ops import (
    BackupRequest,
    BackupResponse,
    CliCommandDecisionRequest,
    CliCommandRequest,
    CliCommandRequestResponse,
    CliCommandUnrestrictedModeRequest,
    CliCommandUnrestrictedModeResponse,
    CliSandboxResponse,
    InitResponse,
    SandboxExecRequest,
    SandboxExecResponse,
)
from app.services.approval_service import ApprovalService, DANGEROUS_MODE_CONFIRMATION
from app.services.backup_service import BackupService
from app.services.cli_sandbox import CliSandboxError, CliSandboxRunner
from app.services.init_service import InitService
from app.services.sandbox_runner import LocalSandboxRunner, SandboxExecutionRecord, SandboxRunnerError

router = APIRouter(tags=["ops"])


async def _run_unrestricted_command(command: str, args: list[str], timeout_seconds: int) -> SandboxExecutionRecord:
    started = datetime.now(timezone.utc).isoformat()
    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        exit_code = int(process.returncode or 0)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise HTTPException(status_code=400, detail="Unrestricted command timed out") from exc
    return SandboxExecutionRecord(
        command=command,
        args=args,
        exit_code=exit_code,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        started_at=started,
        duration_ms=0,
        timeout_seconds=timeout_seconds,
    )


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


@router.post("/ops/cli/request", response_model=CliCommandRequestResponse)
async def cli_request(
    request: CliCommandRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> CliCommandRequestResponse:
    settings = get_settings()
    approval_service = ApprovalService(settings.database_url.get_secret_value())

    unrestricted_enabled, _ = await approval_service.get_cli_unrestricted_mode(subject=principal.subject)
    if unrestricted_enabled:
        record = await _run_unrestricted_command(request.command, request.args, request.timeout_seconds)
        return CliCommandRequestResponse(status="completed", record=record.to_dict())

    has_permission = await approval_service.has_cli_command_permission(subject=principal.subject, command=request.command)
    if settings.cli_sandbox_enabled and has_permission:
        runner = CliSandboxRunner(
            backend=settings.cli_sandbox_backend,
            allowed_commands=settings.sandbox_allowed_commands_list(),
        )
        try:
            record = runner.run(command=request.command, args=request.args, timeout_seconds=request.timeout_seconds)
        except CliSandboxError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CliCommandRequestResponse(status="completed", record=record.to_dict())

    pending = await approval_service.create_pending(
        approval_id=uuid4(),
        action_class="local_sensitive",
        target_host="local",
        tool_name="cli.execute",
        action_payload={
            "command": request.command,
            "args": request.args,
            "timeout_seconds": request.timeout_seconds,
            "requested_by": principal.subject,
        },
        requested_by=principal.subject,
    )
    return CliCommandRequestResponse(status="approval_required", approval_id=str(pending.id))


@router.post("/ops/cli/approvals/{approval_id}/decision", response_model=CliCommandRequestResponse)
async def cli_approval_decision(
    approval_id: UUID,
    request: CliCommandDecisionRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> CliCommandRequestResponse:
    settings = get_settings()
    approval_service = ApprovalService(settings.database_url.get_secret_value())
    approval = await approval_service.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    if approval.tool_name != "cli.execute":
        raise HTTPException(status_code=400, detail="Approval is not a CLI command request")

    if request.decision == "reject":
        await approval_service.reject(approval_id=approval_id, actor=principal.subject, reason="Rejected by user")
        return CliCommandRequestResponse(status="approval_required", approval_id=str(approval_id))

    payload = approval.action_payload
    command = str(payload.get("command", ""))
    args = payload.get("args", [])
    timeout_seconds = int(payload.get("timeout_seconds", 10))
    if not command or not isinstance(args, list):
        raise HTTPException(status_code=400, detail="Approval payload missing CLI command details")

    if request.decision == "allow_always":
        await approval_service.set_cli_command_permission(subject=principal.subject, command=command, actor=principal.subject)

    approved, token, _ = await approval_service.approve(approval_id=approval_id, actor=principal.subject, ttl_minutes=5)
    await approval_service.execute(
        approval_id=approval_id,
        actor=principal.subject,
        execution_token=token,
        expected_payload_hash=approved.payload_hash,
    )

    runner = CliSandboxRunner(
        backend=settings.cli_sandbox_backend,
        allowed_commands=settings.sandbox_allowed_commands_list(),
    )
    try:
        record = runner.run(command=command, args=args, timeout_seconds=timeout_seconds)
    except CliSandboxError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CliCommandRequestResponse(status="completed", record=record.to_dict())


@router.get("/ops/cli/unrestricted", response_model=CliCommandUnrestrictedModeResponse)
async def get_cli_unrestricted_mode(
    principal: Principal = Depends(require_roles("admin", "user")),
) -> CliCommandUnrestrictedModeResponse:
    settings = get_settings()
    service = ApprovalService(settings.database_url.get_secret_value())
    enabled, updated_at = await service.get_cli_unrestricted_mode(subject=principal.subject)
    return CliCommandUnrestrictedModeResponse(enabled=enabled, updated_at=updated_at)


@router.put("/ops/cli/unrestricted", response_model=CliCommandUnrestrictedModeResponse)
async def update_cli_unrestricted_mode(
    request: CliCommandUnrestrictedModeRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> CliCommandUnrestrictedModeResponse:
    settings = get_settings()
    if request.enabled and (
        not request.acknowledged_danger or request.confirmation_text.strip() != DANGEROUS_MODE_CONFIRMATION
    ):
        raise HTTPException(
            status_code=400,
            detail="Dangerous mode requires explicit confirmation and acknowledgement",
        )
    service = ApprovalService(settings.database_url.get_secret_value())
    updated_at = await service.set_cli_unrestricted_mode(
        subject=principal.subject,
        enabled=request.enabled,
        actor=principal.subject,
    )
    return CliCommandUnrestrictedModeResponse(enabled=request.enabled, updated_at=updated_at)
