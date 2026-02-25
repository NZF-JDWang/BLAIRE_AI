"""Outbound notifications."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from blaire_core.config import AppConfig
from blaire_core.telegram_client import send_telegram_audio, send_telegram_document, send_telegram_message, send_telegram_voice

logger = logging.getLogger(__name__)


def _outbox_path(config: AppConfig) -> Path:
    return Path(config.paths.data_root) / "outbox.log"


def notify_user(config: AppConfig, message: str, *, level: str = "info") -> bool:
    """Always log outbound message locally; optionally fan out to Telegram."""
    now = datetime.now(timezone.utc)
    outbox = _outbox_path(config)
    outbox.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": now.isoformat(),
        "level": level,
        "message": message,
    }
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not config.telegram.enabled:
        return True
    if not config.telegram.bot_token or not config.telegram.chat_id:
        logger.warning("telegram: enabled but missing bot token or chat id; skipping send")
        return False

    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[BLAIRE][{level.upper()}][{stamp}]\n{message}"
    return send_telegram_message(
        bot_token=config.telegram.bot_token,
        chat_id=config.telegram.chat_id,
        text=text,
    )


def notify_user_media(
    config: AppConfig,
    file_path: str,
    *,
    media_type: str = "document",
    caption: str | None = None,
    level: str = "info",
) -> bool:
    now = datetime.now(timezone.utc)
    outbox = _outbox_path(config)
    outbox.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": now.isoformat(),
        "level": level,
        "message": f"media:{media_type}",
        "file_path": file_path,
        "caption": caption,
    }
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not config.telegram.enabled:
        return True
    if not config.telegram.bot_token or not config.telegram.chat_id:
        logger.warning("telegram: enabled but missing bot token or chat id; skipping media send")
        return False

    if media_type == "voice":
        return send_telegram_voice(config.telegram.bot_token, config.telegram.chat_id, file_path, caption=caption)
    if media_type == "audio":
        return send_telegram_audio(config.telegram.bot_token, config.telegram.chat_id, file_path, caption=caption)
    return send_telegram_document(config.telegram.bot_token, config.telegram.chat_id, file_path, caption=caption)
