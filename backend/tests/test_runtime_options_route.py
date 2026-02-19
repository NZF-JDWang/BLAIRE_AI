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
    os.environ["MODEL_ALLOW_ANY_INFERENCE"] = "true"
    os.environ["MODEL_ALLOWLIST_EXTRA_GENERAL"] = ""
    os.environ["MODEL_ALLOWLIST_EXTRA_VISION"] = ""
    os.environ["MODEL_ALLOWLIST_EXTRA_EMBEDDING"] = ""
    os.environ["MODEL_ALLOWLIST_EXTRA_CODE"] = ""
    os.environ["MODEL_DISALLOWLIST"] = ""
    os.environ["SEARCH_MODE_DEFAULT"] = "searxng_only"
    os.environ["REQUIRE_AUTH"] = "true"
    os.environ["USER_API_KEYS"] = "test-user-key"
    os.environ["ADMIN_API_KEYS"] = "test-admin-key"


_set_required_env()

from app.main import create_app  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.models.runtime_config import RuntimeConfigEffective  # noqa: E402


def _client() -> TestClient:
    get_settings.cache_clear()
    return TestClient(create_app())


def test_runtime_options_exposes_dynamic_available_models(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.model_router.fetch_available_model_names",
        lambda *_args, **_kwargs: [
            "gpt-oss:20b",
            "dolphin-llama3:8b-v2.9-q4_K_M",
            "nomic-embed-text:v1.5",
            "qwen2.5vl:7b",
        ],
    )
    client = _client()
    response = client.get("/runtime/options", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_search_mode"] == "searxng_only"
    assert "searxng_only" in payload["search_modes"]
    assert "brave_only" in payload["search_modes"]
    assert "gpt-oss:20b" in payload["available_models"]
    assert "dolphin-llama3:8b-v2.9-q4_K_M" in payload["available_models"]


def test_models_endpoint_returns_installed_and_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.model_router.fetch_available_model_names",
        lambda *_args, **_kwargs: ["dolphin-llama3:8b-v2.9-q4_K_M"],
    )
    client = _client()
    response = client.get("/models", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_allow_any_inference"] is True
    assert payload["model_allow_any_inference"] is True
    assert "dolphin-llama3:8b-v2.9-q4_K_M" in payload["installed_models"]
    assert "dolphin-llama3:8b-v2.9-q4_K_M" in payload["allowlist"]["general"]


def test_runtime_options_uses_runtime_config_override(monkeypatch) -> None:
    async def fake_effective(self, settings):  # noqa: ANN001, ANN202
        _ = (self, settings)
        return RuntimeConfigEffective(
            search_mode_default="parallel",
            sensitive_actions_enabled=False,
            approval_token_ttl_minutes=22,
            allowed_network_hosts=["example.local"],
            allowed_network_tools=["network_probe"],
            allowed_obsidian_paths=["notes"],
            allowed_ha_operations=["light.turn_on"],
            allowed_homelab_operations=["dns.resolve"],
        )

    monkeypatch.setattr("app.services.runtime_config_service.RuntimeConfigService.get_effective", fake_effective)
    client = _client()
    response = client.get("/runtime/options", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_search_mode"] == "parallel"
    assert payload["sensitive_actions_enabled"] is False
    assert payload["approval_token_ttl_minutes"] == 22


def test_runtime_diagnostics_returns_effective_config(monkeypatch) -> None:
    async def fake_effective(self, settings):  # noqa: ANN001, ANN202
        _ = (self, settings)
        return RuntimeConfigEffective(
            search_mode_default="parallel",
            sensitive_actions_enabled=False,
            approval_token_ttl_minutes=19,
            allowed_network_hosts=["example.local"],
            allowed_network_tools=["network_probe"],
            allowed_obsidian_paths=["notes"],
            allowed_ha_operations=["light.turn_on"],
            allowed_homelab_operations=["dns.resolve"],
        )

    monkeypatch.setattr("app.services.runtime_config_service.RuntimeConfigService.get_effective", fake_effective)
    client = _client()
    response = client.get("/runtime/diagnostics", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "user"
    assert payload["effective_search_mode_default"] == "parallel"
    assert payload["effective_sensitive_actions_enabled"] is False
    assert payload["effective_approval_token_ttl_minutes"] == 19


def test_runtime_system_summary_admin_only() -> None:
    client = _client()
    user_response = client.get("/runtime/system-summary", headers={"X-API-Key": "test-user-key"})
    assert user_response.status_code == 403

    admin_response = client.get("/runtime/system-summary", headers={"X-API-Key": "test-admin-key"})
    assert admin_response.status_code == 200
    payload = admin_response.json()
    assert payload["app_env"] in {"production", "development", "staging", "test"}
    assert payload["restart_required_note"].startswith("These values come from environment")
