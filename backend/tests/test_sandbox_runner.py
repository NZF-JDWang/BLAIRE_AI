import os

import pytest
from fastapi.testclient import TestClient

from app.services.sandbox_runner import LocalSandboxRunner, SandboxExecutionRecord, SandboxRunnerError


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
    os.environ["ADMIN_API_KEYS"] = "test-admin-key"
    os.environ["USER_API_KEYS"] = "test-user-key"
    os.environ["SANDBOX_ALLOWED_COMMANDS"] = "echo"


@pytest.mark.anyio
async def test_runner_blocks_non_allowlisted_command() -> None:
    runner = LocalSandboxRunner(["echo"])
    with pytest.raises(SandboxRunnerError, match="not allowlisted"):
        await runner.run("python", ["-V"])


def test_sandbox_execute_route(monkeypatch) -> None:
    _set_required_env()
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()

    async def fake_run(self, command: str, args: list[str], timeout_seconds: int = 10):  # noqa: ANN001, ANN202
        _ = (self, command, args, timeout_seconds)
        return SandboxExecutionRecord(
            command="echo",
            args=["hello"],
            exit_code=0,
            stdout="hello\n",
            stderr="",
            started_at="2026-02-18T00:00:00+00:00",
            duration_ms=3.2,
            timeout_seconds=10,
        )

    monkeypatch.setattr(LocalSandboxRunner, "run", fake_run)
    client = TestClient(create_app())
    response = client.post(
        "/ops/sandbox/execute",
        headers={"X-API-Key": "test-admin-key"},
        json={"command": "echo", "args": ["hello"], "timeout_seconds": 10},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["record"]["command"] == "echo"
    assert payload["record"]["exit_code"] == 0

