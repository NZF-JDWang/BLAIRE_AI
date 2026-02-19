from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class CalendarEventsResponse(BaseModel):
    events: list[dict[str, Any]]


class GmailSendRequest(BaseModel):
    to: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=20000)
    approval_id: UUID | None = None
    execution_token: str | None = Field(default=None, min_length=20, max_length=256)
    expected_payload_hash: str | None = Field(default=None, min_length=64, max_length=64)


class IntegrationActionResponse(BaseModel):
    status: str
    source: str
    payload_hash: str | None = None
    approval_id: UUID | None = None
    data: dict[str, Any] | None = None


class ImapMessagesResponse(BaseModel):
    messages: list[dict[str, Any]]


class IntegrationsStatusResponse(BaseModel):
    google_oauth_configured: bool
    google_api_base: str
    imap_configured: bool
    imap_host: str
    home_assistant_configured: bool
    home_assistant_url: str
