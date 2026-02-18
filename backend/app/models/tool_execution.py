from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

ExecutionStatus = Literal["completed", "approval_required"]
ActionClass = Literal["local_safe", "local_sensitive", "network_sensitive"]


class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = Field(default="system", min_length=1, max_length=128)
    target_host: str | None = Field(default=None, max_length=255)
    approval_id: UUID | None = None
    execution_token: str | None = Field(default=None, min_length=20, max_length=256)
    expected_payload_hash: str | None = Field(default=None, min_length=64, max_length=64)


class ToolExecutionResult(BaseModel):
    tool_name: str
    action_class: ActionClass
    status: ExecutionStatus
    output: dict[str, Any] | None = None
    approval_id: UUID | None = None
    payload_hash: str | None = None

