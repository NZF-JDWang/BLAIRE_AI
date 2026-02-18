import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models.preferences import PreferenceResponse


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


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.preferences_service import PreferencesService  # noqa: E402


def _fake_pref() -> PreferenceResponse:
    return PreferenceResponse(
        subject="user",
        search_mode="searxng_only",
        model_class="general",
        model_override=None,
        updated_at=datetime.now(timezone.utc),
    )


def test_get_preferences(monkeypatch) -> None:
    async def fake_get_or_default(self, *, subject: str, default_search_mode: str, default_model_class: str = "general"):  # noqa: ANN001, ANN202
        _ = (self, subject, default_search_mode, default_model_class)
        return _fake_pref()

    monkeypatch.setattr(PreferencesService, "get_or_default", fake_get_or_default)
    client = TestClient(create_app())
    response = client.get("/preferences/me", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    assert response.json()["search_mode"] == "searxng_only"


def test_update_preferences(monkeypatch) -> None:
    async def fake_upsert(self, *, subject: str, request):  # noqa: ANN001, ANN202
        _ = (self, subject, request)
        return PreferenceResponse(
            subject="user",
            search_mode="parallel",
            model_class="general",
            model_override=None,
            updated_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(PreferencesService, "upsert", fake_upsert)
    client = TestClient(create_app())
    response = client.put(
        "/preferences/me",
        headers={"X-API-Key": "test-user-key"},
        json={"search_mode": "parallel", "model_class": "general", "model_override": None},
    )
    assert response.status_code == 200
    assert response.json()["search_mode"] == "parallel"


def test_update_preferences_rejects_disallowed_override(monkeypatch) -> None:
    async def fail_if_called(self, *, subject: str, request):  # noqa: ANN001, ANN202
        _ = (self, subject, request)
        raise AssertionError("upsert should not be called for disallowed overrides")

    monkeypatch.setattr(PreferencesService, "upsert", fail_if_called)
    client = TestClient(create_app())
    response = client.put(
        "/preferences/me",
        headers={"X-API-Key": "test-user-key"},
        json={
            "search_mode": "parallel",
            "model_class": "general",
            "model_override": "not-allowed-model",
        },
    )
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]
