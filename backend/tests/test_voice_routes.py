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
    os.environ["REQUIRE_AUTH"] = "true"
    os.environ["USER_API_KEYS"] = "test-user-key"
    os.environ["ADMIN_API_KEYS"] = "test-admin-key"


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.voice_service import VoiceService  # noqa: E402


def test_tts_route(monkeypatch) -> None:
    def fake_tts(self, text: str):  # noqa: ANN001, ANN202
        _ = (self, text)
        return ("ZmFrZQ==", 22050)

    monkeypatch.setattr(VoiceService, "synthesize_tts", fake_tts)
    client = TestClient(create_app())
    response = client.post("/voice/tts", headers={"X-API-Key": "test-user-key"}, json={"text": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_base64"] == "ZmFrZQ=="


def test_stt_route(monkeypatch) -> None:
    def fake_stt(self, audio_path):  # noqa: ANN001, ANN202
        _ = (self, audio_path)
        return "hello world"

    monkeypatch.setattr(VoiceService, "transcribe_stt", fake_stt)
    client = TestClient(create_app())
    response = client.post(
        "/voice/stt",
        headers={"X-API-Key": "test-user-key"},
        files={"file": ("voice.wav", b"fake-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello world"

