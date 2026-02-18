from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

ApprovalStatus = Literal["pending", "approved", "rejected", "executed", "expired"]
ActionClass = Literal["local_safe", "local_sensitive", "network_sensitive"]


class ApprovalCreateRequest(BaseModel):
    action_class: ActionClass
    target_host: str = Field(min_length=1, max_length=255)
    tool_name: str = Field(min_length=1, max_length=128)
    action_payload: dict[str, Any]
    requested_by: str = Field(default="system", min_length=1, max_length=128)


class ApprovalRecord(BaseModel):
    id: UUID
    status: ApprovalStatus
    action_class: ActionClass
    target_host: str
    tool_name: str
    action_payload: dict[str, Any]
    payload_hash: str
    requested_by: str
    approved_by: str | None
    created_at: datetime
    updated_at: datetime
    token_expires_at: datetime | None
    executed_at: datetime | None
    rejection_reason: str | None


class ApprovalCreateResponse(BaseModel):
    approval: ApprovalRecord


class ApprovalActionRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=128)


class ApprovalRejectRequest(ApprovalActionRequest):
    reason: str = Field(min_length=1, max_length=500)


class ApprovalApproveResponse(BaseModel):
    approval: ApprovalRecord
    execution_token: str
    expires_at: datetime


class ApprovalExecuteRequest(ApprovalActionRequest):
    execution_token: str = Field(min_length=20, max_length=256)
    expected_payload_hash: str = Field(min_length=64, max_length=64)


class ApprovalAuditEvent(BaseModel):
    id: int
    approval_id: UUID | None
    event_type: str
    actor: str
    details: dict[str, Any]
    event_time: datetime
