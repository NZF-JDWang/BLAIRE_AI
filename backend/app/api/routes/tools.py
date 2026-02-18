from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.core.rate_limit import RateLimitRule, rate_limiter
from app.models.tool_execution import ToolExecutionRequest, ToolExecutionResult
from app.services.approval_service import ApprovalService, canonical_payload_hash
from app.services.tool_policy import ToolPolicy, ToolPolicyError
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["tools"])


@router.get("/tools")
async def list_tools(_: Principal = Depends(require_roles("admin", "user"))) -> list[dict[str, str | bool]]:
    registry = ToolRegistry()
    return [
        {
            "name": spec.name,
            "action_class": spec.action_class,
            "description": spec.description,
            "requires_target_host": spec.requires_target_host,
        }
        for spec in registry.list_specs()
    ]


@router.post("/tools/execute", response_model=ToolExecutionResult)
async def execute_tool(
    request: ToolExecutionRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> ToolExecutionResult:
    settings = get_settings()
    registry = ToolRegistry()
    approval_service = ApprovalService(settings.database_url.get_secret_value())
    policy = ToolPolicy(settings)

    tool = registry.get(request.tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")

    spec = tool.spec
    if spec.requires_target_host and not request.target_host:
        raise HTTPException(status_code=400, detail="target_host is required for this tool")

    try:
        policy.validate_network_tool(spec, request.target_host)
    except ToolPolicyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    payload = {
        "tool_name": spec.name,
        "arguments": request.arguments,
        "target_host": request.target_host,
    }
    payload_hash = canonical_payload_hash(payload)

    rate_limiter.check(
        f"tool:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}",
        RateLimitRule(60, 60),
    )

    if spec.action_class == "network_sensitive":
        if not request.approval_id or not request.execution_token or not request.expected_payload_hash:
            record = await approval_service.create_pending(
                approval_id=uuid4(),
                action_class="network_sensitive",
                target_host=request.target_host or "",
                tool_name=spec.name,
                action_payload=payload,
                requested_by=principal.subject,
            )
            return ToolExecutionResult(
                tool_name=spec.name,
                action_class=spec.action_class,
                status="approval_required",
                approval_id=record.id,
                payload_hash=record.payload_hash,
            )

        if request.expected_payload_hash != payload_hash:
            raise HTTPException(status_code=409, detail="Payload hash does not match request payload")

        try:
            await approval_service.execute(
                approval_id=request.approval_id,
                actor=principal.subject,
                execution_token=request.execution_token,
                expected_payload_hash=request.expected_payload_hash,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    output = await tool.run(request.arguments, request.target_host)
    return ToolExecutionResult(
        tool_name=spec.name,
        action_class=spec.action_class,
        status="completed",
        output=output,
    )
