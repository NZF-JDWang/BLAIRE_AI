from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.core.rate_limit import RateLimitRule, rate_limiter
from app.models.integrations import (
    CalendarEventsResponse,
    GmailSendRequest,
    ImapMessagesResponse,
    IntegrationActionResponse,
)
from app.services.approval_service import ApprovalService, canonical_payload_hash
from app.services.integrations_service import GoogleIntegrationService, ImapIntegrationService, IntegrationServiceError

router = APIRouter(tags=["integrations"])


def _google_service() -> GoogleIntegrationService:
    settings = get_settings()
    token = settings.google_oauth_token.get_secret_value() if settings.google_oauth_token else ""
    if not token:
        raise HTTPException(status_code=503, detail="Google OAuth token is not configured")
    return GoogleIntegrationService(api_base=settings.google_api_base, oauth_token=token)


@router.get("/integrations/google/calendar/events", response_model=CalendarEventsResponse)
async def google_calendar_events(
    calendar_id: str = "primary",
    max_results: int = 10,
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> CalendarEventsResponse:
    try:
        events = await _google_service().list_calendar_events(calendar_id=calendar_id, max_results=max(1, min(max_results, 20)))
    except IntegrationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return CalendarEventsResponse(events=events)


@router.post("/integrations/google/gmail/send", response_model=IntegrationActionResponse)
async def gmail_send(
    request: GmailSendRequest,
    raw_request: Request,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> IntegrationActionResponse:
    settings = get_settings()
    rate_limiter.check(
        f"gmail-send:{principal.subject}:{raw_request.client.host if raw_request.client else 'unknown'}",
        RateLimitRule(20, 60),
    )
    if not settings.sensitive_actions_enabled:
        raise HTTPException(status_code=503, detail="Sensitive actions are globally disabled")

    approval_service = ApprovalService(settings.database_url.get_secret_value())
    payload = {"to": request.to, "subject": request.subject, "body": request.body, "operation": "gmail.send"}
    payload_hash = canonical_payload_hash(payload)

    if not request.approval_id or not request.execution_token or not request.expected_payload_hash:
        record = await approval_service.create_pending(
            approval_id=uuid4(),
            action_class="network_sensitive",
            target_host="gmail_api",
            tool_name="gmail_send",
            action_payload=payload,
            requested_by=principal.subject,
        )
        return IntegrationActionResponse(
            status="approval_required",
            source="gmail",
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

    try:
        result = await _google_service().send_gmail(to=request.to, subject=request.subject, body=request.body)
    except IntegrationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return IntegrationActionResponse(status="completed", source="gmail", data=result)


@router.get("/integrations/imap/messages", response_model=ImapMessagesResponse)
async def imap_messages(
    limit: int = 10,
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> ImapMessagesResponse:
    settings = get_settings()
    password = settings.imap_password.get_secret_value() if settings.imap_password else ""
    if not settings.imap_host or not settings.imap_user or not password:
        raise HTTPException(status_code=503, detail="IMAP credentials are not configured")
    service = ImapIntegrationService(host=settings.imap_host, username=settings.imap_user, password=password)
    try:
        messages = service.list_recent_messages(limit=max(1, min(limit, 50)))
    except IntegrationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ImapMessagesResponse(messages=messages)
