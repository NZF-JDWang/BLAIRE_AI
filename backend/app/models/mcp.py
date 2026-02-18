from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

McpStatus = Literal["completed", "approval_required"]


class ObsidianReadRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


class ObsidianWriteRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    content: str = Field(max_length=200000)
    requested_by: str = Field(default="system", min_length=1, max_length=128)
    approval_id: UUID | None = None
    execution_token: str | None = Field(default=None, min_length=20, max_length=256)
    expected_payload_hash: str | None = Field(default=None, min_length=64, max_length=64)


class HomeAssistantCallRequest(BaseModel):
    operation: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = Field(default="system", min_length=1, max_length=128)
    approval_id: UUID | None = None
    execution_token: str | None = Field(default=None, min_length=20, max_length=256)
    expected_payload_hash: str | None = Field(default=None, min_length=64, max_length=64)


class McpActionResponse(BaseModel):
    status: McpStatus
    source: str
    payload_hash: str | None = None
    approval_id: UUID | None = None
    data: dict[str, Any] | None = None

