import os
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient


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
    os.environ["CLI_SANDBOX_ENABLED"] = "true"
    os.environ["CLI_SANDBOX_BACKEND"] = "firejail"
    os.environ["SANDBOX_ALLOWED_COMMANDS"] = "echo"


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.approval_service import ApprovalService  # noqa: E402
from app.services.cli_sandbox import CliSandboxRunner  # noqa: E402
from app.services.init_service import InitService  # noqa: E402


def test_ops_init_route(monkeypatch) -> None:
    async def fake_run(self):  # noqa: ANN001, ANN202
        _ = self
        return {
            "approval_schema_ready": True,
            "preferences_schema_ready": True,
            "metadata_schema_ready": True,
            "qdrant_collection_ready": True,
        }

    monkeypatch.setattr(InitService, "run", fake_run)
    client = TestClient(create_app())
    response = client.post("/ops/init", headers={"X-API-Key": "test-admin-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["steps"]["metadata_schema_ready"] is True


def test_ops_cli_execute_route(monkeypatch) -> None:
    def fake_run(self, *, command: str, args: list[str], timeout_seconds: int = 10):  # noqa: ANN001, ANN202
        _ = (self, command, args, timeout_seconds)
        from app.services.cli_sandbox import CliSandboxRecord

        return CliSandboxRecord(
            command="echo",
            args=["hello"],
            backend="firejail",
            exit_code=0,
            stdout="hello\n",
            stderr="",
            started_at="2026-02-18T00:00:00+00:00",
            timeout_seconds=10,
        )

    monkeypatch.setattr(CliSandboxRunner, "run", fake_run)
    client = TestClient(create_app())
    response = client.post(
        "/ops/cli/execute",
        headers={"X-API-Key": "test-admin-key"},
        json={"command": "echo", "args": ["hello"], "timeout_seconds": 10},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["record"]["backend"] == "firejail"


def test_cli_request_creates_approval_when_not_allowlisted(monkeypatch) -> None:
    async def fake_get_mode(self, *, subject: str):  # noqa: ANN001, ANN202
        _ = (self, subject)
        return False, datetime.now(timezone.utc)

    async def fake_has_permission(self, *, subject: str, command: str):  # noqa: ANN001, ANN202
        _ = (self, subject, command)
        return False

    async def fake_create_pending(self, **kwargs):  # noqa: ANN001, ANN202
        return SimpleNamespace(id=uuid4(), **kwargs)

    monkeypatch.setattr(ApprovalService, "get_cli_unrestricted_mode", fake_get_mode)
    monkeypatch.setattr(ApprovalService, "has_cli_command_permission", fake_has_permission)
    monkeypatch.setattr(ApprovalService, "create_pending", fake_create_pending)

    client = TestClient(create_app())
    response = client.post(
        "/ops/cli/request",
        headers={"X-API-Key": "test-user-key"},
        json={"command": "echo", "args": ["hello"], "timeout_seconds": 10},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "approval_required"
    assert payload["approval_id"]


def test_update_unrestricted_mode_requires_confirmation() -> None:
    client = TestClient(create_app())
    response = client.put(
        "/ops/cli/unrestricted",
        headers={"X-API-Key": "test-user-key"},
        json={"enabled": True, "acknowledged_danger": False, "confirmation_text": "nope"},
    )
    assert response.status_code == 400
