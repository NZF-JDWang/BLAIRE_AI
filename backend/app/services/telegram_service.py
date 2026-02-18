from typing import Any

import httpx


class TelegramServiceError(RuntimeError):
    pass


class TelegramService:
    def __init__(self, bot_token: str):
        self._base = f"https://api.telegram.org/bot{bot_token}"

    async def send_message(self, *, chat_id: str, text: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._base}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise TelegramServiceError("Telegram sendMessage failed") from exc
        if not isinstance(data, dict) or not data.get("ok", False):
            raise TelegramServiceError("Telegram API returned non-ok response")
        return data

    @staticmethod
    def parse_incoming_text(update: dict[str, Any]) -> tuple[str, str] | None:
        message = update.get("message") or {}
        text = message.get("text")
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not text or chat_id is None:
            return None
        return str(chat_id), str(text)
