import os

from fastapi.testclient import TestClient

from app.models.runtime_config import RuntimeConfigBundle, RuntimeConfigEffective, RuntimeConfigOverrides


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
from app.core.config import get_settings  # noqa: E402
from app.services.runtime_config_service import RuntimeConfigService  # noqa: E402


def _client() -> TestClient:
    get_settings.cache_clear()
    return TestClient(create_app())


def _bundle() -> RuntimeConfigBundle:
    return RuntimeConfigBundle(
        effective=RuntimeConfigEffective(
            search_mode_default="parallel",
            sensitive_actions_enabled=True,
            approval_token_ttl_minutes=15,
            allowed_network_hosts=["example.local"],
            allowed_network_tools=["network_probe"],
            allowed_obsidian_paths=["notes"],
            allowed_ha_operations=["light.turn_on"],
            allowed_homelab_operations=["dns.resolve"],
        ),
        overrides=RuntimeConfigOverrides(
            search_mode_default="parallel",
            sensitive_actions_enabled=True,
            approval_token_ttl_minutes=15,
            allowed_network_hosts="example.local",
            allowed_network_tools="network_probe",
            allowed_obsidian_paths="notes",
            allowed_ha_operations="light.turn_on",
            allowed_homelab_operations="dns.resolve",
            updated_by="admin",
            updated_at=None,
        ),
    )


def test_runtime_config_admin_only(monkeypatch) -> None:
    async def fake_bundle(self, settings):  # noqa: ANN001, ANN202
        _ = (self, settings)
        return _bundle()

    monkeypatch.setattr(RuntimeConfigService, "get_bundle", fake_bundle)
    client = _client()
    response = client.get("/runtime/config", headers={"X-API-Key": "test-user-key"})
    assert response.status_code == 403


def test_runtime_config_get_as_admin(monkeypatch) -> None:
    async def fake_bundle(self, settings):  # noqa: ANN001, ANN202
        _ = (self, settings)
        return _bundle()

    monkeypatch.setattr(RuntimeConfigService, "get_bundle", fake_bundle)
    client = _client()
    response = client.get("/runtime/config", headers={"X-API-Key": "test-admin-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["effective"]["search_mode_default"] == "parallel"
    assert payload["overrides"]["allowed_network_tools"] == "network_probe"


def test_runtime_config_update_as_admin(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_upsert(self, *, actor: str, request):  # noqa: ANN001, ANN202
        _ = self
        calls.append(actor)
        assert request.search_mode_default == "parallel"
        return _bundle().overrides

    async def fake_bundle(self, settings):  # noqa: ANN001, ANN202
        _ = (self, settings)
        return _bundle()

    monkeypatch.setattr(RuntimeConfigService, "upsert", fake_upsert)
    monkeypatch.setattr(RuntimeConfigService, "get_bundle", fake_bundle)
    client = _client()
    response = client.put(
        "/runtime/config",
        headers={"X-API-Key": "test-admin-key"},
        json={
            "search_mode_default": "parallel",
            "sensitive_actions_enabled": True,
            "approval_token_ttl_minutes": 15,
            "allowed_network_hosts": "example.local",
            "allowed_network_tools": "network_probe",
            "allowed_obsidian_paths": "notes",
            "allowed_ha_operations": "light.turn_on",
            "allowed_homelab_operations": "dns.resolve",
        },
    )
    assert response.status_code == 200
    assert calls == ["admin"]
