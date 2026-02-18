from uuid import uuid4
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.core.rate_limit import RateLimitRule, rate_limiter
from app.models.mcp import (
    HomeAssistantCallRequest,
    HomelabCallRequest,
    McpActionResponse,
    ObsidianReadRequest,
    ObsidianWriteRequest,
)
from app.services.approval_service import ApprovalService, canonical_payload_hash
from app.services.mcp_client import McpClient, McpClientError

router = APIRouter(tags=["mcp"])


def _normalize_obsidian_path(path: str) -> str:
    candidate = path.strip().replace("\\", "/")
    if not candidate:
        raise HTTPException(status_code=400, detail="Path cannot be empty")
    normalized = str(PurePosixPath(candidate))
    if normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise HTTPException(status_code=400, detail="Path traversal is not allowed")
    if normalized.startswith("/"):
        raise HTTPException(status_code=400, detail="Absolute paths are not allowed")
    return normalized


def _enforce_obsidian_scope(path: str) -> str:
    settings = get_settings()
    normalized = _normalize_obsidian_path(path)
    allowed = settings.allowed_obsidian_paths_list()
    if not allowed:
        return normalized
    for prefix in allowed:
        cleaned = str(PurePosixPath(prefix.strip().replace("\\", "/")))
        if not cleaned:
            continue
        if normalized == cleaned or normalized.startswith(cleaned.rstrip("/") + "/"):
            return normalized
    raise HTTPException(status_code=403, detail="Obsidian path is not allowlisted")


def _mcp_envelope(
    *,
    source: str,
    operation: str,
    request_payload: dict,
    status: str,
    result: dict | None = None,
    approval_id: str | None = None,
    payload_hash: str | None = None,
) -> dict:
    return {
        "request": {
            "source": source,
            "operation": operation,
            "payload": request_payload,
        },
        "result": result or {},
        "audit": {
            "status": status,
            "approval_id": approval_id,
            "payload_hash": payload_hash,
        },
    }


@router.post("/mcp/obsidian/read", response_model=McpActionResponse)
async def obsidian_read(
    request: ObsidianReadRequest,
    _: Principal = Depends(require_roles("admin", "user")),
) -> McpActionResponse:
    settings = get_settings()
    client = McpClient()
    scoped_path = _enforce_obsidian_scope(request.path)
    try:
        data = await client.call(settings.mcp_obsidian_url, "vault.read", {"path": scoped_path})
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return McpActionResponse(
        status="completed",
        source="obsidian",
        data=data,
        envelope=_mcp_envelope(
            source="obsidian",
            operation="vault.read",
            request_payload={"path": scoped_path},
            status="completed",
            result=data,
        ),
    )


@router.post("/mcp/obsidian/write", response_model=McpActionResponse)
async def obsidian_write(
    request: ObsidianWriteRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> McpActionResponse:
    settings = get_settings()
    scoped_path = _enforce_obsidian_scope(request.path)
    if not settings.sensitive_actions_enabled:
        raise HTTPException(status_code=503, detail="Sensitive actions are globally disabled")
    rate_limiter.check(
        f"mcp-obsidian:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}",
        RateLimitRule(30, 60),
    )
    approval_service = ApprovalService(settings.database_url.get_secret_value())
    payload = {"path": scoped_path, "content": request.content, "operation": "vault.write"}
    payload_hash = canonical_payload_hash(payload)

    if not request.approval_id or not request.execution_token or not request.expected_payload_hash:
        record = await approval_service.create_pending(
            approval_id=uuid4(),
            action_class="network_sensitive",
            target_host="obsidian_mcp",
            tool_name="mcp_obsidian_write",
            action_payload=payload,
            requested_by=principal.subject,
        )
        return McpActionResponse(
            status="approval_required",
            source="obsidian",
            approval_id=record.id,
            payload_hash=record.payload_hash,
            envelope=_mcp_envelope(
                source="obsidian",
                operation="vault.write",
                request_payload=payload,
                status="approval_required",
                approval_id=str(record.id),
                payload_hash=record.payload_hash,
            ),
        )

    if request.expected_payload_hash != payload_hash:
        raise HTTPException(status_code=409, detail="Payload hash mismatch")

    try:
        await approval_service.execute(
            approval_id=request.approval_id,
            actor=principal.subject,
            execution_token=request.execution_token,
            expected_payload_hash=request.expected_payload_hash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    client = McpClient()
    try:
        data = await client.call(settings.mcp_obsidian_url, "vault.write", payload)
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return McpActionResponse(
        status="completed",
        source="obsidian",
        data=data,
        envelope=_mcp_envelope(
            source="obsidian",
            operation="vault.write",
            request_payload=payload,
            status="completed",
            result=data,
        ),
    )


@router.post("/mcp/ha/call", response_model=McpActionResponse)
async def home_assistant_call(
    request: HomeAssistantCallRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> McpActionResponse:
    settings = get_settings()
    if not settings.sensitive_actions_enabled:
        raise HTTPException(status_code=503, detail="Sensitive actions are globally disabled")
    rate_limiter.check(
        f"mcp-ha:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}",
        RateLimitRule(30, 60),
    )
    allowed = settings.allowed_ha_operations_list()
    if allowed and request.operation not in allowed:
        raise HTTPException(status_code=403, detail="Home Assistant operation is not allowlisted")

    approval_service = ApprovalService(settings.database_url.get_secret_value())
    payload = {"operation": request.operation, "payload": request.payload}
    payload_hash = canonical_payload_hash(payload)

    if not request.approval_id or not request.execution_token or not request.expected_payload_hash:
        record = await approval_service.create_pending(
            approval_id=uuid4(),
            action_class="network_sensitive",
            target_host="ha_mcp",
            tool_name="mcp_ha_call",
            action_payload=payload,
            requested_by=principal.subject,
        )
        return McpActionResponse(
            status="approval_required",
            source="home_assistant",
            approval_id=record.id,
            payload_hash=record.payload_hash,
            envelope=_mcp_envelope(
                source="home_assistant",
                operation=request.operation,
                request_payload=payload,
                status="approval_required",
                approval_id=str(record.id),
                payload_hash=record.payload_hash,
            ),
        )

    if request.expected_payload_hash != payload_hash:
        raise HTTPException(status_code=409, detail="Payload hash mismatch")

    try:
        await approval_service.execute(
            approval_id=request.approval_id,
            actor=principal.subject,
            execution_token=request.execution_token,
            expected_payload_hash=request.expected_payload_hash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    client = McpClient()
    try:
        data = await client.call(
            settings.mcp_ha_url,
            "ha.call_service",
            {"operation": request.operation, "payload": request.payload},
        )
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return McpActionResponse(
        status="completed",
        source="home_assistant",
        data=data,
        envelope=_mcp_envelope(
            source="home_assistant",
            operation=request.operation,
            request_payload={"operation": request.operation, "payload": request.payload},
            status="completed",
            result=data,
        ),
    )


@router.post("/mcp/homelab/call", response_model=McpActionResponse)
async def homelab_call(
    request: HomelabCallRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> McpActionResponse:
    settings = get_settings()
    if not settings.sensitive_actions_enabled:
        raise HTTPException(status_code=503, detail="Sensitive actions are globally disabled")
    rate_limiter.check(
        f"mcp-homelab:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}",
        RateLimitRule(30, 60),
    )
    allowed = settings.allowed_homelab_operations_list()
    if allowed and request.operation not in allowed:
        raise HTTPException(status_code=403, detail="Homelab operation is not allowlisted")

    approval_service = ApprovalService(settings.database_url.get_secret_value())
    payload = {"operation": request.operation, "payload": request.payload}
    payload_hash = canonical_payload_hash(payload)

    if not request.approval_id or not request.execution_token or not request.expected_payload_hash:
        record = await approval_service.create_pending(
            approval_id=uuid4(),
            action_class="network_sensitive",
            target_host="homelab_mcp",
            tool_name="mcp_homelab_call",
            action_payload=payload,
            requested_by=principal.subject,
        )
        return McpActionResponse(
            status="approval_required",
            source="homelab",
            approval_id=record.id,
            payload_hash=record.payload_hash,
            envelope=_mcp_envelope(
                source="homelab",
                operation=request.operation,
                request_payload=payload,
                status="approval_required",
                approval_id=str(record.id),
                payload_hash=record.payload_hash,
            ),
        )

    if request.expected_payload_hash != payload_hash:
        raise HTTPException(status_code=409, detail="Payload hash mismatch")

    try:
        await approval_service.execute(
            approval_id=request.approval_id,
            actor=principal.subject,
            execution_token=request.execution_token,
            expected_payload_hash=request.expected_payload_hash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    client = McpClient()
    try:
        data = await client.call(
            settings.mcp_homelab_url,
            "homelab.call",
            {"operation": request.operation, "payload": request.payload},
        )
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return McpActionResponse(
        status="completed",
        source="homelab",
        data=data,
        envelope=_mcp_envelope(
            source="homelab",
            operation=request.operation,
            request_payload={"operation": request.operation, "payload": request.payload},
            status="completed",
            result=data,
        ),
    )
