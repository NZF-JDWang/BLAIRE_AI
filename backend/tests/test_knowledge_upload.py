import os
from pathlib import Path

from fastapi.testclient import TestClient


def _set_required_env(drop_folder: Path) -> None:
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
    os.environ["DROP_FOLDER"] = str(drop_folder)


def test_knowledge_upload_saves_file(tmp_path: Path) -> None:
    _set_required_env(tmp_path)
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    client = TestClient(create_app())
    response = client.post(
        "/knowledge/upload",
        headers={"X-API-Key": "test-user-key"},
        files={"file": ("note.md", b"# hello", "text/markdown")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["bytes"] > 0
    stored = tmp_path / payload["stored_filename"]
    assert stored.exists()


