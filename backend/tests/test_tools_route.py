import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models.approval import ApprovalRecord
from app.services.approval_service import canonical_payload_hash


def _set_required_env() -> None:
    os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/db")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("MCP_OBSIDIAN_URL", "http://localhost:3000")
    os.environ.setdefault("MCP_HA_URL", "http://localhost:3001")
    os.environ.setdefault("API_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1,backend")
    os.environ.setdefault("MODEL_GENERAL_DEFAULT", "qwen2.5:7b-instruct")
    os.environ.setdefault("MODEL_VISION_DEFAULT", "qwen2.5vl:7b")
    os.environ.setdefault("MODEL_EMBEDDING_DEFAULT", "nomic-embed-text:v1.5")
    os.environ.setdefault("ALLOWED_NETWORK_HOSTS", "host-a")
    os.environ.setdefault("ALLOWED_NETWORK_TOOLS", "network_probe")


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.approval_service import ApprovalService  # noqa: E402


def test_local_safe_tool_executes() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/tools/execute",
        json={
            "tool_name": "echo_text",
            "arguments": {"text": "hello"},
            "requested_by": "tester",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["output"]["echo"] == "hello"


def test_network_sensitive_returns_approval_required(monkeypatch) -> None:
    async def fake_create_pending(self, **kwargs):  # noqa: ANN001, ANN202
        now = datetime.now(timezone.utc)
        payload_hash = canonical_payload_hash(kwargs["action_payload"])
        return ApprovalRecord(
            id=kwargs["approval_id"],
            status="pending",
            action_class="network_sensitive",
            target_host=kwargs["target_host"],
            tool_name=kwargs["tool_name"],
            action_payload=kwargs["action_payload"],
            payload_hash=payload_hash,
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
        "/tools/execute",
        json={
            "tool_name": "network_probe",
            "arguments": {"check": "status"},
            "target_host": "host-a",
            "requested_by": "tester",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "approval_required"
    assert payload["approval_id"] is not None
    assert payload["payload_hash"] is not None
