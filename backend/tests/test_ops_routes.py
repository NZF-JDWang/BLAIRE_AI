import os

from fastapi.testclient import TestClient


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
    os.environ["CLI_SANDBOX_ENABLED"] = "true"
    os.environ["CLI_SANDBOX_BACKEND"] = "firejail"
    os.environ["SANDBOX_ALLOWED_COMMANDS"] = "echo"


_set_required_env()

from app.main import create_app  # noqa: E402
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

