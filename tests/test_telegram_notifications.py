from __future__ import annotations

import json
from pathlib import Path

import blaire_core.telegram_client as telegram_client
from blaire_core.config import read_config_snapshot
from blaire_core.notifications import notify_user, notify_user_media


def test_config_disables_telegram_if_enabled_without_required_fields(monkeypatch) -> None:
    monkeypatch.setenv("BLAIRE_TELEGRAM_ENABLED", "true")
    monkeypatch.delenv("BLAIRE_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("BLAIRE_TELEGRAM_CHAT_ID", raising=False)

    snapshot = read_config_snapshot("dev")
    assert snapshot.valid
    assert snapshot.warnings
    assert snapshot.effective_config is not None
    assert snapshot.effective_config.telegram.enabled is False


def test_send_telegram_message_builds_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"ok":true}'

    def _fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse()

    monkeypatch.setattr(telegram_client.urllib.request, "urlopen", _fake_urlopen)

    ok = telegram_client.send_telegram_message("token-123", "42", "hello")
    assert ok is True
    assert captured["url"] == "https://api.telegram.org/bottoken-123/sendMessage"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 10
    assert captured["body"] == {"chat_id": "42", "text": "hello", "disable_notification": False}


def test_notify_user_writes_outbox_under_data_root(tmp_path: Path) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path)})
    assert snapshot.effective_config is not None

    sent = notify_user(snapshot.effective_config, "test outbound", level="warn")
    assert sent is True

    outbox = tmp_path / "outbox.log"
    assert outbox.exists()
    entries = outbox.read_text(encoding="utf-8").strip().splitlines()
    assert len(entries) == 1
    payload = json.loads(entries[0])
    assert payload["level"] == "warn"
    assert payload["message"] == "test outbound"


def test_telegram_polling_env_is_disabled_when_telegram_disabled(monkeypatch) -> None:
    monkeypatch.setenv("BLAIRE_TELEGRAM_ENABLED", "false")
    monkeypatch.setenv("BLAIRE_TELEGRAM_POLLING_ENABLED", "true")

    snapshot = read_config_snapshot("dev")
    assert snapshot.effective_config is not None
    assert snapshot.effective_config.telegram.enabled is False
    assert snapshot.effective_config.telegram.polling_enabled is False


def test_send_telegram_file_builds_multipart_request(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")
    captured: dict[str, object] = {}

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"ok":true}'

    def _fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["content_type"] = request.headers.get("Content-type")
        captured["body"] = request.data
        return DummyResponse()

    monkeypatch.setattr(telegram_client.urllib.request, "urlopen", _fake_urlopen)
    assert telegram_client.send_telegram_document("token-abc", "1001", str(file_path), caption="cap") is True
    assert captured["url"] == "https://api.telegram.org/bottoken-abc/sendDocument"
    assert str(captured["content_type"]).startswith("multipart/form-data; boundary=")
    assert b"name=\"chat_id\"" in captured["body"]
    assert b"name=\"document\"; filename=\"note.txt\"" in captured["body"]


def test_notify_user_media_logs_even_when_telegram_disabled(tmp_path: Path) -> None:
    media_file = tmp_path / "a.ogg"
    media_file.write_bytes(b"x")
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path)})
    assert snapshot.effective_config is not None
    ok = notify_user_media(snapshot.effective_config, str(media_file), media_type="voice", caption="memo")
    assert ok is True
    outbox = tmp_path / "outbox.log"
    payload = json.loads(outbox.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["message"] == "media:voice"
