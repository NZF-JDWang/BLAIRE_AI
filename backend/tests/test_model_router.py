import os

from app.core.config import Settings
from app.services.model_router import ModelRouter


def _settings() -> Settings:
    os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/db")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("MCP_OBSIDIAN_URL", "http://localhost:3000")
    os.environ.setdefault("MCP_HA_URL", "http://localhost:3001")
    os.environ.setdefault("MODEL_GENERAL_DEFAULT", "qwen2.5:7b-instruct")
    os.environ.setdefault("MODEL_VISION_DEFAULT", "qwen2.5vl:7b")
    os.environ.setdefault("MODEL_EMBEDDING_DEFAULT", "nomic-embed-text:v1.5")
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
