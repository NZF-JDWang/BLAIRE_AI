import os

from fastapi.testclient import TestClient

from app.models.agent import ResearchResponse, WorkerResult


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
    os.environ["TELEGRAM_BOT_TOKEN"] = "fake-token"
    os.environ["TELEGRAM_WEBHOOK_SECRET_TOKEN"] = "secret-token"


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.agent_swarm import AgentSwarmService  # noqa: E402
from app.services.telegram_service import TelegramService  # noqa: E402


def test_telegram_webhook_ignored_for_non_text() -> None:
    from app.core.config import get_settings

    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/telegram/webhook",
        json={"update_id": 1},
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret-token"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


def test_telegram_webhook_runs_research_and_sends(monkeypatch) -> None:
    from app.core.config import get_settings

    get_settings.cache_clear()

    async def fake_research(self, query: str, search_mode: str | None = None, recursion_depth: int = 0):  # noqa: ANN001, ANN202
        _ = (self, query, search_mode, recursion_depth)
        return ResearchResponse(
            query=query,
            supervisor_summary="summary text",
            workers=[WorkerResult(worker_id="w1", summary="s", sources=["https://a"])],
        )

    async def fake_send(self, *, chat_id: str, text: str):  # noqa: ANN001, ANN202
        _ = (self, chat_id, text)
        return {"ok": True}

    monkeypatch.setattr(AgentSwarmService, "run_research", fake_research)
    monkeypatch.setattr(TelegramService, "send_message", fake_send)
    client = TestClient(create_app())
    response = client.post(
        "/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret-token"},
        json={
            "message": {
                "chat": {"id": 12345},
                "text": "/research backup strategy",
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_telegram_webhook_rejects_invalid_secret() -> None:
    from app.core.config import get_settings

    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        json={"message": {"chat": {"id": 12345}, "text": "hello"}},
    )
    assert response.status_code == 403

