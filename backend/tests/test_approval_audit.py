import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.models.approval import ApprovalAuditEvent


def _set_required_env() -> None:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["INFERENCE_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["API_ALLOWED_HOSTS"] = "testserver,localhost,127.0.0.1,backend"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["REQUIRE_AUTH"] = "true"
    os.environ["USER_API_KEYS"] = "test-user-key"
    os.environ["ADMIN_API_KEYS"] = "test-admin-key"


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.approval_service import ApprovalService  # noqa: E402


def test_approval_audit_route(monkeypatch) -> None:
    aid = uuid4()

    async def fake_list_audit_events(self, approval_id, limit=200):  # noqa: ANN001, ANN202
        _ = (self, approval_id, limit)
        return [
            ApprovalAuditEvent(
                id=1,
                approval_id=aid,
                event_type="approved",
                actor="admin",
                details={},
                event_time=datetime.now(timezone.utc),
            )
        ]

    monkeypatch.setattr(ApprovalService, "list_audit_events", fake_list_audit_events)
    client = TestClient(create_app())
    response = client.get(f"/approvals/{aid}/audit", headers={"X-API-Key": "test-admin-key"})
    assert response.status_code == 200
    assert response.json()[0]["event_type"] == "approved"


