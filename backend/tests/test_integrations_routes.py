import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models.approval import ApprovalRecord


def _set_required_env() -> None:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["API_ALLOWED_HOSTS"] = "testserver,localhost,127.0.0.1,backend"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["REQUIRE_AUTH"] = "true"
    os.environ["USER_API_KEYS"] = "test-user-key"
    os.environ["ADMIN_API_KEYS"] = "test-admin-key"
    os.environ["GOOGLE_OAUTH_TOKEN"] = "fake-google-token"
    os.environ["IMAP_HOST"] = "imap.test.local"
    os.environ["IMAP_USER"] = "user@test.local"
    os.environ["IMAP_PASSWORD"] = "password"


_set_required_env()

from app.core.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.approval_service import ApprovalService  # noqa: E402
from app.services.integrations_service import GoogleIntegrationService, ImapIntegrationService  # noqa: E402


def test_google_calendar_events(monkeypatch) -> None:
    get_settings.cache_clear()

    async def fake_events(self, calendar_id: str = "primary", max_results: int = 10):  # noqa: ANN001, ANN202
        _ = (self, calendar_id, max_results)
        return [{"id": "evt-1", "summary": "Meeting"}]

    monkeypatch.setattr(GoogleIntegrationService, "list_calendar_events", fake_events)
    client = TestClient(create_app())
    response = client.get("/integrations/google/calendar/events", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    assert response.json()["events"][0]["id"] == "evt-1"


def test_gmail_send_requires_approval(monkeypatch) -> None:
    get_settings.cache_clear()

    async def fake_create_pending(self, **kwargs):  # noqa: ANN001, ANN202
        now = datetime.now(timezone.utc)
        return ApprovalRecord(
            id=kwargs["approval_id"],
            status="pending",
            action_class="network_sensitive",
            target_host="gmail_api",
            tool_name="gmail_send",
            action_payload=kwargs["action_payload"],
            payload_hash="c" * 64,
            requested_by=kwargs["requested_by"],
            approved_by=None,
            created_at=now,
            updated_at=now,
            token_expires_at=None,
            executed_at=None,
            rejection_reason=None,
        )

    monkeypatch.setattr(ApprovalService, "create_pending", fake_create_pending)
    client = TestClient(create_app())
    response = client.post(
        "/integrations/google/gmail/send",
        headers={"X-API-Key": "test-user-key"},
        json={"to": "a@test.com", "subject": "hi", "body": "hello"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approval_required"


def test_imap_messages(monkeypatch) -> None:
    get_settings.cache_clear()

    def fake_messages(self, limit: int = 10):  # noqa: ANN001, ANN202
        _ = (self, limit)
        return [{"id": "1", "subject": "hello"}]

    monkeypatch.setattr(ImapIntegrationService, "list_recent_messages", fake_messages)
    client = TestClient(create_app())
    response = client.get("/integrations/imap/messages", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    assert response.json()["messages"][0]["subject"] == "hello"
