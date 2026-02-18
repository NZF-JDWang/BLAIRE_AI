import os

from app.core.config import Settings
from app.services.model_router import ModelRouter


def _settings(*, allow_any_inference: bool = False) -> Settings:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["MODEL_ALLOW_ANY_INFERENCE"] = "true" if allow_any_inference else "false"
    os.environ["MODEL_ALLOWLIST_EXTRA_GENERAL"] = ""
    os.environ["MODEL_ALLOWLIST_EXTRA_VISION"] = ""
    os.environ["MODEL_ALLOWLIST_EXTRA_EMBEDDING"] = ""
    os.environ["MODEL_ALLOWLIST_EXTRA_CODE"] = ""
    os.environ["MODEL_DISALLOWLIST"] = ""
    return Settings()


def test_select_model_uses_session_override_when_allowed() -> None:
    router = ModelRouter(_settings())
    selection = router.select_model("general", request_override="llama3.2:3b")

    assert selection.model_name == "llama3.2:3b"
    assert selection.reason == "session_override"
    assert selection.fallback_used is False
    assert selection.rejected_candidates == []


def test_select_model_falls_back_to_preference_override() -> None:
    router = ModelRouter(_settings())
    selection = router.select_model(
        "general",
        request_override="not-allowed-model",
        preference_override="llama3.2:3b",
    )

    assert selection.model_name == "llama3.2:3b"
    assert selection.reason == "user_preference_override"
    assert selection.fallback_used is True
    assert selection.rejected_candidates == ["session_override_disallowed:not-allowed-model"]


def test_select_model_falls_back_to_class_default() -> None:
    router = ModelRouter(_settings())
    selection = router.select_model(
        "general",
        request_override="bad-session-override",
        preference_override="bad-preference-override",
    )

    assert selection.model_name == "qwen2.5:7b-instruct"
    assert selection.reason == "class_default"
    assert selection.fallback_used is True
    assert selection.rejected_candidates == [
        "session_override_disallowed:bad-session-override",
        "preference_override_disallowed:bad-preference-override",
    ]


def test_select_model_allows_installed_override_when_allow_any_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.model_router.fetch_available_model_names",
        lambda *_args, **_kwargs: ["dolphin-llama3:8b-v2.9-q4_K_M"],
    )
    router = ModelRouter(_settings(allow_any_inference=True))
    selection = router.select_model("general", request_override="dolphin-llama3:8b-v2.9-q4_K_M")
    assert selection.model_name == "dolphin-llama3:8b-v2.9-q4_K_M"
    assert selection.reason == "session_override"


def test_select_model_rejects_installed_override_when_allow_any_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.model_router.fetch_available_model_names",
        lambda *_args, **_kwargs: ["dolphin-llama3:8b-v2.9-q4_K_M"],
    )
    router = ModelRouter(_settings(allow_any_inference=False))
    selection = router.select_model("general", request_override="dolphin-llama3:8b-v2.9-q4_K_M")
    assert selection.model_name == "qwen2.5:7b-instruct"
    assert selection.reason == "class_default"
    assert selection.rejected_candidates == ["session_override_disallowed:dolphin-llama3:8b-v2.9-q4_K_M"]
