from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import blaire_core.telegram_bridge as telegram_bridge
from blaire_core.config import read_config_snapshot


def _build_context(tmp_path: Path, *, enabled: bool = True, polling_enabled: bool = True) -> SimpleNamespace:
    snapshot = read_config_snapshot(
        "dev",
        {
            "paths.data_root": str(tmp_path),
            "telegram.enabled": str(enabled).lower(),
            "telegram.bot_token": "token-123",
            "telegram.chat_id": "42",
            "telegram.polling_enabled": str(polling_enabled).lower(),
            "telegram.polling_timeout_seconds": "1",
        },
    )
    assert snapshot.effective_config is not None
    return SimpleNamespace(config=snapshot.effective_config)


def test_process_telegram_updates_handles_text_and_replies(monkeypatch, tmp_path: Path) -> None:
    context = _build_context(tmp_path)
    calls: dict[str, list[object]] = {"handle": [], "notify": []}

    def _fake_handle(context_obj, session_id: str, user_message: str) -> str:
        calls["handle"].append((session_id, user_message))
        return "assistant reply"

    def _fake_notify(config_obj, message: str, *, level: str = "info", via_telegram: bool = False) -> bool:
        _ = config_obj
        calls["notify"].append((message, level, via_telegram))
        return True

    monkeypatch.setattr(telegram_bridge, "handle_user_message", _fake_handle)
    monkeypatch.setattr(telegram_bridge, "notify_user", _fake_notify)

    updates = [
        {
            "update_id": 1,
            "message": {
                "chat": {"id": 42},
                "from": {"id": 100, "is_bot": False},
                "text": "hello there",
            },
        }
    ]
    handled = telegram_bridge.process_telegram_updates(context, updates)

    assert handled == 1
    assert calls["handle"] == [("telegram-42", "hello there")]
    assert calls["notify"] == [("assistant reply", "info", True)]


def test_process_telegram_updates_ignores_other_chats_and_bots(monkeypatch, tmp_path: Path) -> None:
    context = _build_context(tmp_path)
    calls: dict[str, int] = {"handle": 0, "notify": 0}

    def _fake_handle(context_obj, session_id: str, user_message: str) -> str:
        _ = (context_obj, session_id, user_message)
        calls["handle"] += 1
        return "assistant reply"

    def _fake_notify(config_obj, message: str, *, level: str = "info", via_telegram: bool = False) -> bool:
        _ = (config_obj, message, level, via_telegram)
        calls["notify"] += 1
        return True

    monkeypatch.setattr(telegram_bridge, "handle_user_message", _fake_handle)
    monkeypatch.setattr(telegram_bridge, "notify_user", _fake_notify)

    updates = [
        {"update_id": 1, "message": {"chat": {"id": 777}, "from": {"is_bot": False}, "text": "wrong chat"}},
        {"update_id": 2, "message": {"chat": {"id": 42}, "from": {"is_bot": True}, "text": "bot text"}},
    ]
    handled = telegram_bridge.process_telegram_updates(context, updates)

    assert handled == 0
    assert calls["handle"] == 0
    assert calls["notify"] == 0


def test_telegram_bridge_persists_offset(tmp_path: Path) -> None:
    context = _build_context(tmp_path)
    bridge = telegram_bridge.TelegramTextBridge(context)

    bridge._offset = 55
    bridge._save_offset()

    state_file = tmp_path / "telegram_offset.json"
    assert state_file.exists()
    assert json.loads(state_file.read_text(encoding="utf-8")) == {"offset": 55}

    bridge2 = telegram_bridge.TelegramTextBridge(context)
    assert bridge2.status().offset == 55
