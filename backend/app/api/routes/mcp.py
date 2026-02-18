from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.core.rate_limit import RateLimitRule, rate_limiter
from app.models.mcp import (
    HomeAssistantCallRequest,
    McpActionResponse,
    ObsidianReadRequest,
    ObsidianWriteRequest,
)
from app.services.approval_service import ApprovalService, canonical_payload_hash
from app.services.mcp_client import McpClient, McpClientError

router = APIRouter(tags=["mcp"])


@router.post("/mcp/obsidian/read", response_model=McpActionResponse)
async def obsidian_read(
    request: ObsidianReadRequest,
    _: Principal = Depends(require_roles("admin", "user")),
) -> McpActionResponse:
    settings = get_settings()
    client = McpClient()
    try:
        data = await client.call(settings.mcp_obsidian_url, "vault.read", {"path": request.path})
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return McpActionResponse(status="completed", source="obsidian", data=data)


@router.post("/mcp/obsidian/write", response_model=McpActionResponse)
async def obsidian_write(
    request: ObsidianWriteRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> McpActionResponse:
    settings = get_settings()
    if not settings.sensitive_actions_enabled:
        raise HTTPException(status_code=503, detail="Sensitive actions are globally disabled")
    rate_limiter.check(
        f"mcp-obsidian:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}",
        RateLimitRule(30, 60),
    )
    approval_service = ApprovalService(settings.database_url.get_secret_value())
    payload = {"path": request.path, "content": request.content, "operation": "vault.write"}
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
    return McpActionResponse(status="completed", source="obsidian", data=data)


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
    return McpActionResponse(status="completed", source="home_assistant", data=data)
