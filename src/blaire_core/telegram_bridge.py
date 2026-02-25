"""Telegram text polling bridge."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from blaire_core.notifications import notify_user
from blaire_core.orchestrator import handle_user_message
from blaire_core.telegram_client import get_telegram_updates

if TYPE_CHECKING:
    from blaire_core.orchestrator import AppContext


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TelegramBridgeStatus:
    running: bool
    polling_enabled: bool
    offset: int | None
    last_error: str | None


def process_telegram_updates(context: AppContext, updates: list[dict[str, object]]) -> int:
    """Process inbound Telegram updates for configured chat."""
    telegram = context.config.telegram
    target_chat_id = str(telegram.chat_id or "")
    handled = 0

    for update in updates:
        if not isinstance(update, dict):
            continue
        message = update.get("message")
        if not isinstance(message, dict):
            continue
        chat = message.get("chat")
        if not isinstance(chat, dict):
            continue
        if str(chat.get("id", "")) != target_chat_id:
            continue
        sender = message.get("from")
        if isinstance(sender, dict) and bool(sender.get("is_bot")):
            continue

        session_id = f"telegram-{target_chat_id}"
        text = message.get("text")
        if isinstance(text, str) and text.strip():
            try:
                reply = handle_user_message(context, session_id=session_id, user_message=text.strip())
            except Exception as exc:  # noqa: BLE001
                logger.exception("telegram: inbound text handling failed")
                notify_user(context.config, f"Inbound Telegram processing failed: {exc}", level="error", via_telegram=True)
            else:
                notify_user(context.config, reply, level="info", via_telegram=True)
            handled += 1
            continue

        if "voice" in message or "audio" in message or "document" in message:
            notify_user(
                context.config,
                "Received media file. Media ingestion is acknowledged.",
                level="info",
                via_telegram=True,
            )
            handled += 1

    return handled


class TelegramTextBridge:
    """Background Telegram long-polling bridge for text chat."""

    def __init__(self, context: AppContext) -> None:
        self._context = context
        self._offset: int | None = self._load_offset()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_error: str | None = None

    def _offset_path(self) -> Path:
        return Path(self._context.config.paths.data_root) / "telegram_offset.json"

    def _load_offset(self) -> int | None:
        path = self._offset_path()
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if isinstance(raw, dict) and isinstance(raw.get("offset"), int):
            return int(raw["offset"])
        return None

    def _save_offset(self) -> None:
        path = self._offset_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"offset": self._offset}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def start(self) -> bool:
        telegram = self._context.config.telegram
        if not telegram.polling_enabled:
            return False
        if not telegram.enabled or not telegram.bot_token or not telegram.chat_id:
            self._last_error = "telegram disabled or missing credentials"
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()

        def _runner() -> None:
            logger.info("telegram bridge: polling started")
            while not self._stop.is_set():
                try:
                    updates, next_offset = get_telegram_updates(
                        telegram.bot_token or "",
                        offset=self._offset,
                        timeout=telegram.polling_timeout_seconds,
                    )
                    if next_offset is not None and next_offset != self._offset:
                        self._offset = next_offset
                        self._save_offset()
                    if updates:
                        process_telegram_updates(self._context, updates)
                    self._last_error = None
                except Exception as exc:  # noqa: BLE001
                    self._last_error = str(exc)
                    logger.exception("telegram bridge: polling loop failed")
                    # Avoid hot-looping when network is down.
                    time.sleep(2.0)
            logger.info("telegram bridge: polling stopped")

        self._thread = threading.Thread(target=_runner, name="blaire-telegram-poller", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def status(self) -> TelegramBridgeStatus:
        return TelegramBridgeStatus(
            running=bool(self._thread and self._thread.is_alive()),
            polling_enabled=bool(self._context.config.telegram.polling_enabled),
            offset=self._offset,
            last_error=self._last_error,
        )
