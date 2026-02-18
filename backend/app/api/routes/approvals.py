from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.core.rate_limit import RateLimitRule, rate_limiter
from app.models.approval import (
    ApprovalApproveResponse,
    ApprovalCreateRequest,
    ApprovalCreateResponse,
    ApprovalExecuteRequest,
    ApprovalRecord,
    ApprovalRejectRequest,
)
from app.services.approval_service import ApprovalService

router = APIRouter(tags=["approvals"])


def _service() -> ApprovalService:
    settings = get_settings()
    return ApprovalService(settings.database_url.get_secret_value())


@router.get("/approvals/pending", response_model=list[ApprovalRecord])
async def list_pending_approvals(
    limit: int = Query(default=50, ge=1, le=200),
    _: Principal = Depends(require_roles("admin")),
) -> list[ApprovalRecord]:
    return await _service().list_pending(limit=limit)


@router.post("/approvals", response_model=ApprovalCreateResponse, status_code=201)
async def create_approval(
    request: ApprovalCreateRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> ApprovalCreateResponse:
    record = await _service().create_pending(
        approval_id=uuid4(),
        action_class=request.action_class,
        target_host=request.target_host,
        tool_name=request.tool_name,
        action_payload=request.action_payload,
        requested_by=principal.subject,
    )
    return ApprovalCreateResponse(approval=record)


@router.get("/approvals/{approval_id}", response_model=ApprovalRecord)
async def get_approval(approval_id: UUID, _: Principal = Depends(require_roles("admin"))) -> ApprovalRecord:
    record = await _service().get_approval(approval_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return record


@router.post("/approvals/{approval_id}/approve", response_model=ApprovalApproveResponse)
async def approve_approval(
    approval_id: UUID,
    request: Request,
    principal: Principal = Depends(require_roles("admin")),
) -> ApprovalApproveResponse:
    settings = get_settings()
    rate_limiter.check(f"approve:{principal.subject}:{request.client.host if request.client else 'unknown'}", RateLimitRule(20, 60))
    try:
        record, token, expires_at = await _service().approve(
            approval_id=approval_id,
            actor=principal.subject,
            ttl_minutes=settings.approval_token_ttl_minutes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ApprovalApproveResponse(approval=record, execution_token=token, expires_at=expires_at)


@router.post("/approvals/{approval_id}/reject", response_model=ApprovalRecord)
async def reject_approval(
    approval_id: UUID,
    request: ApprovalRejectRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin")),
) -> ApprovalRecord:
    rate_limiter.check(f"reject:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}", RateLimitRule(20, 60))
    try:
        return await _service().reject(approval_id=approval_id, actor=principal.subject, reason=request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/approvals/{approval_id}/execute", response_model=ApprovalRecord)
async def execute_approval(
    approval_id: UUID,
    request: ApprovalExecuteRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin")),
) -> ApprovalRecord:
    settings = get_settings()
    if not settings.sensitive_actions_enabled:
        raise HTTPException(status_code=503, detail="Sensitive actions are globally disabled")
    rate_limiter.check(f"execute:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}", RateLimitRule(30, 60))
    try:
        return await _service().execute(
            approval_id=approval_id,
            actor=principal.subject,
            execution_token=request.execution_token,
            expected_payload_hash=request.expected_payload_hash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
