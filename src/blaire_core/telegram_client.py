"""Telegram client helpers for outbound and polling."""

from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from pathlib import Path
import urllib.error
import urllib.request


logger = logging.getLogger(__name__)
_MAX_TELEGRAM_TEXT_CHARS = 4000


def _chunk_text(text: str, max_chars: int = _MAX_TELEGRAM_TEXT_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n")
    return [chunk for chunk in chunks if chunk]


def _telegram_api_request(bot_token: str, method: str, payload: dict[str, object], *, timeout: int = 10) -> tuple[bool, dict[str, object] | None]:
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    request = urllib.request.Request(
        url=url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
            if not 200 <= response.status < 300:
                logger.error("telegram: %s failed status=%s body=%s", method, response.status, body)
                return False, None
            return True, data if isinstance(data, dict) else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("telegram: %s failed status=%s body=%s", method, exc.code, body)
    except Exception as exc:  # noqa: BLE001
        logger.error("telegram: %s failed error=%s", method, exc)
    return False, None


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str | None = None,
    disable_notification: bool = False,
) -> bool:
    chunks = _chunk_text(text)
    logger.info("telegram: sending message to chat_id=%s (len=%s, chunks=%s)", chat_id, len(text), len(chunks))
    for chunk in chunks:
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_notification": disable_notification,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        ok, _ = _telegram_api_request(bot_token, "sendMessage", payload)
        if not ok:
            return False
    return True


def _multipart_payload(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----blaire-{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for key, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    lines.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(lines), boundary


def send_telegram_file(bot_token: str, method: str, field_name: str, chat_id: str, file_path: str, *, caption: str | None = None) -> bool:
    path = Path(file_path)
    if not path.exists():
        logger.error("telegram: file does not exist path=%s", file_path)
        return False
    fields = {"chat_id": chat_id}
    if caption:
        fields["caption"] = caption
    body, boundary = _multipart_payload(fields, field_name, path)
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    request = urllib.request.Request(
        url=url,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        data=body,
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            _ = response.read()
            if not 200 <= response.status < 300:
                logger.error("telegram: %s failed status=%s", method, response.status)
                return False
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("telegram: %s failed error=%s", method, exc)
        return False


def send_telegram_voice(bot_token: str, chat_id: str, file_path: str, *, caption: str | None = None) -> bool:
    return send_telegram_file(bot_token, "sendVoice", "voice", chat_id, file_path, caption=caption)


def send_telegram_audio(bot_token: str, chat_id: str, file_path: str, *, caption: str | None = None) -> bool:
    return send_telegram_file(bot_token, "sendAudio", "audio", chat_id, file_path, caption=caption)


def send_telegram_document(bot_token: str, chat_id: str, file_path: str, *, caption: str | None = None) -> bool:
    return send_telegram_file(bot_token, "sendDocument", "document", chat_id, file_path, caption=caption)


def get_telegram_updates(bot_token: str, *, offset: int | None = None, timeout: int = 20) -> tuple[list[dict[str, object]], int | None]:
    payload: dict[str, object] = {"timeout": timeout}
    if offset is not None:
        payload["offset"] = offset
    ok, data = _telegram_api_request(bot_token, "getUpdates", payload, timeout=timeout + 5)
    if not ok or not data:
        return [], offset
    results = data.get("result", [])
    if not isinstance(results, list):
        return [], offset
    updates = [item for item in results if isinstance(item, dict)]
    if not updates:
        return [], offset
    next_offset = max(int(item.get("update_id", 0)) for item in updates) + 1
    return updates, next_offset
