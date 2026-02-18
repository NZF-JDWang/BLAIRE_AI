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
    os.environ["ALLOWED_OBSIDIAN_PATHS"] = "notes,daily"
    os.environ["ALLOWED_HA_OPERATIONS"] = "light.turn_on"


_set_required_env()

from app.main import create_app  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.approval_service import ApprovalService  # noqa: E402
from app.services.mcp_client import McpClient  # noqa: E402


def test_obsidian_read_works(monkeypatch) -> None:
    get_settings.cache_clear()
    async def fake_call(self, base_url: str, method: str, params: dict):  # noqa: ANN001, ANN202
        return {"ok": True, "method": method, "params": params}

    monkeypatch.setattr(McpClient, "call", fake_call)
    client = TestClient(create_app())
    response = client.post("/mcp/obsidian/read", json={"path": "notes/a.md"}, headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["source"] == "obsidian"
    assert payload["envelope"]["request"]["operation"] == "vault.read"


def test_obsidian_write_requires_approval(monkeypatch) -> None:
    get_settings.cache_clear()
    async def fake_create_pending(self, **kwargs):  # noqa: ANN001, ANN202
        now = datetime.now(timezone.utc)
        return ApprovalRecord(
            id=kwargs["approval_id"],
            status="pending",
            action_class="network_sensitive",
            target_host="obsidian_mcp",
            tool_name="mcp_obsidian_write",
            action_payload=kwargs["action_payload"],
            payload_hash="a" * 64,
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
        "/mcp/obsidian/write",
        json={"path": "notes/a.md", "content": "x"},
        headers={"X-API-Key": "test-user-key"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "approval_required"
    assert payload["envelope"]["audit"]["status"] == "approval_required"


def test_home_assistant_allowlist_enforced() -> None:
    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/mcp/ha/call",
        json={"operation": "switch.turn_off", "payload": {}},
        headers={"X-API-Key": "test-user-key"},
    )
    assert response.status_code == 403


def test_obsidian_read_scope_enforced(monkeypatch) -> None:
    get_settings.cache_clear()

    async def fake_call(self, base_url: str, method: str, params: dict):  # noqa: ANN001, ANN202
        _ = (self, base_url, method, params)
        return {"ok": True}

    monkeypatch.setattr(McpClient, "call", fake_call)
    client = TestClient(create_app())
    response = client.post(
        "/mcp/obsidian/read",
        json={"path": "private/secrets.md"},
        headers={"X-API-Key": "test-user-key"},
    )
    assert response.status_code == 403


def test_obsidian_read_rejects_path_traversal(monkeypatch) -> None:
    get_settings.cache_clear()

    async def fake_call(self, base_url: str, method: str, params: dict):  # noqa: ANN001, ANN202
        _ = (self, base_url, method, params)
        return {"ok": True}

    monkeypatch.setattr(McpClient, "call", fake_call)
    client = TestClient(create_app())
    response = client.post(
        "/mcp/obsidian/read",
        json={"path": "../secret.md"},
        headers={"X-API-Key": "test-user-key"},
    )
    assert response.status_code == 400
