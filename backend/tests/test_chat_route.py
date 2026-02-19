import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models.preferences import PreferenceResponse


def _set_required_env() -> None:
    os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/db")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("INFERENCE_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("MCP_OBSIDIAN_URL", "http://localhost:3000")
    os.environ.setdefault("MCP_HA_URL", "http://localhost:3001")
    os.environ.setdefault("API_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1,backend")
    os.environ.setdefault("MODEL_GENERAL_DEFAULT", "qwen2.5:7b-instruct")
    os.environ.setdefault("MODEL_VISION_DEFAULT", "qwen2.5vl:7b")
    os.environ.setdefault("MODEL_EMBEDDING_DEFAULT", "nomic-embed-text:v1.5")
    os.environ.setdefault("REQUIRE_AUTH", "true")
    os.environ.setdefault("USER_API_KEYS", "test-user-key")
    os.environ.setdefault("ADMIN_API_KEYS", "test-admin-key")


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.inference_client import InferenceClient  # noqa: E402
from app.services.preferences_service import PreferencesService  # noqa: E402


def _fake_pref() -> PreferenceResponse:
    return PreferenceResponse(
        subject="user",
        search_mode="searxng_only",
        model_class="general",
        model_override=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=None,
        context_window_tokens=None,
        use_rag=True,
        retrieval_k=4,
        updated_at=datetime.now(timezone.utc),
    )


def test_chat_streaming_response(monkeypatch) -> None:
    async def fake_get_or_default(self, *, subject: str, default_search_mode: str, default_model_class: str = "general"):  # noqa: ANN001, ANN202
        _ = (self, subject, default_search_mode, default_model_class)
        return _fake_pref()

    async def fake_stream_chat(self, model: str, messages: list[dict[str, str]], **kwargs):  # noqa: ANN001, ANN202
        _ = (self, model, messages, kwargs)
        yield "Hello"
        yield " world"

    monkeypatch.setattr(PreferencesService, "get_or_default", fake_get_or_default)
    monkeypatch.setattr(InferenceClient, "stream_chat", fake_stream_chat)

    client = TestClient(create_app())
    response = client.post(
        "/chat",
        headers={"X-API-Key": "test-user-key"},
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "use_rag": False,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: meta" in response.text
    assert '"model": "qwen2.5:7b-instruct"' in response.text
    assert '"fallback_used": false' in response.text
    assert "event: token" in response.text
    assert '"text": "Hello world"' in response.text


def test_chat_non_streaming_response(monkeypatch) -> None:
    async def fake_get_or_default(self, *, subject: str, default_search_mode: str, default_model_class: str = "general"):  # noqa: ANN001, ANN202
        _ = (self, subject, default_search_mode, default_model_class)
        return _fake_pref()

    async def fake_stream_chat(self, model: str, messages: list[dict[str, str]], **kwargs):  # noqa: ANN001, ANN202
        _ = (self, model, messages, kwargs)
        yield "Hello"
        yield " world"

    monkeypatch.setattr(PreferencesService, "get_or_default", fake_get_or_default)
    monkeypatch.setattr(InferenceClient, "stream_chat", fake_stream_chat)

    client = TestClient(create_app())
    response = client.post(
        "/chat",
        headers={"X-API-Key": "test-user-key"},
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False,
            "use_rag": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "qwen2.5:7b-instruct"
    assert payload["text"] == "Hello world"
    assert payload["citations"] == []
    assert payload["rag_status"] == "disabled"
    assert payload["rag_error"] is None


def test_chat_invalid_session_override_falls_back(monkeypatch) -> None:
    async def fake_get_or_default(self, *, subject: str, default_search_mode: str, default_model_class: str = "general"):  # noqa: ANN001, ANN202
        _ = (self, subject, default_search_mode, default_model_class)
        return _fake_pref()

    async def fake_stream_chat(self, model: str, messages: list[dict[str, str]], **kwargs):  # noqa: ANN001, ANN202
        _ = (self, model, messages, kwargs)
        yield "fallback ok"

    monkeypatch.setattr(PreferencesService, "get_or_default", fake_get_or_default)
    monkeypatch.setattr(InferenceClient, "stream_chat", fake_stream_chat)

    client = TestClient(create_app())
    response = client.post(
        "/chat",
        headers={"X-API-Key": "test-user-key"},
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "model_override": "not-allowed-model",
            "stream": False,
            "use_rag": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "qwen2.5:7b-instruct"
    assert payload["text"] == "fallback ok"

