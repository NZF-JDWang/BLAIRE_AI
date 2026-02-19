import os

from fastapi.testclient import TestClient


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


_set_required_env()

from app.main import create_app  # noqa: E402


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "blaire-backend"

