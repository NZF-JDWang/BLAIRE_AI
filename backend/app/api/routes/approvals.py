from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.models.approval import (
    ApprovalActionRequest,
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
async def list_pending_approvals(limit: int = Query(default=50, ge=1, le=200)) -> list[ApprovalRecord]:
    return await _service().list_pending(limit=limit)


@router.post("/approvals", response_model=ApprovalCreateResponse, status_code=201)
async def create_approval(request: ApprovalCreateRequest) -> ApprovalCreateResponse:
    record = await _service().create_pending(
        approval_id=uuid4(),
        action_class=request.action_class,
        target_host=request.target_host,
        tool_name=request.tool_name,
        action_payload=request.action_payload,
        requested_by=request.requested_by,
    )
    return ApprovalCreateResponse(approval=record)


@router.get("/approvals/{approval_id}", response_model=ApprovalRecord)
async def get_approval(approval_id: UUID) -> ApprovalRecord:
    record = await _service().get_approval(approval_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return record


@router.post("/approvals/{approval_id}/approve", response_model=ApprovalApproveResponse)
async def approve_approval(approval_id: UUID, request: ApprovalActionRequest) -> ApprovalApproveResponse:
    settings = get_settings()
    try:
        record, token, expires_at = await _service().approve(
            approval_id=approval_id,
            actor=request.actor,
            ttl_minutes=settings.approval_token_ttl_minutes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ApprovalApproveResponse(approval=record, execution_token=token, expires_at=expires_at)


@router.post("/approvals/{approval_id}/reject", response_model=ApprovalRecord)
async def reject_approval(approval_id: UUID, request: ApprovalRejectRequest) -> ApprovalRecord:
    try:
        return await _service().reject(approval_id=approval_id, actor=request.actor, reason=request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/approvals/{approval_id}/execute", response_model=ApprovalRecord)
async def execute_approval(approval_id: UUID, request: ApprovalExecuteRequest) -> ApprovalRecord:
    settings = get_settings()
    if not settings.sensitive_actions_enabled:
        raise HTTPException(status_code=503, detail="Sensitive actions are globally disabled")
    try:
        return await _service().execute(
            approval_id=approval_id,
            actor=request.actor,
            execution_token=request.execution_token,
            expected_payload_hash=request.expected_payload_hash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

