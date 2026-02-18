import base64
import email.utils
import imaplib
from typing import Any

import httpx


class IntegrationServiceError(RuntimeError):
    pass


class GoogleIntegrationService:
    def __init__(self, *, api_base: str, oauth_token: str):
        self._api_base = api_base.rstrip("/")
        self._token = oauth_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def list_calendar_events(self, calendar_id: str = "primary", max_results: int = 10) -> list[dict[str, Any]]:
        url = f"{self._api_base}/calendar/v3/calendars/{calendar_id}/events"
        params = {"maxResults": max_results, "singleEvents": "true", "orderBy": "startTime"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=self._headers(), params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise IntegrationServiceError("Google Calendar request failed") from exc
        items = data.get("items", [])
        return items if isinstance(items, list) else []

    async def send_gmail(self, *, to: str, subject: str, body: str) -> dict[str, Any]:
        url = f"{self._api_base}/gmail/v1/users/me/messages/send"
        raw_message = f"To: {to}\r\nSubject: {subject}\r\n\r\n{body}"
        raw_encoded = base64.urlsafe_b64encode(raw_message.encode("utf-8")).decode("ascii")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=self._headers(), json={"raw": raw_encoded})
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise IntegrationServiceError("Gmail send failed") from exc
        return data if isinstance(data, dict) else {}


class ImapIntegrationService:
    def __init__(self, *, host: str, username: str, password: str):
        self._host = host
        self._username = username
        self._password = password

    def list_recent_messages(self, limit: int = 10) -> list[dict[str, Any]]:
        try:
            with imaplib.IMAP4_SSL(self._host) as client:
                client.login(self._username, self._password)
                client.select("INBOX")
                status, data = client.search(None, "ALL")
                if status != "OK" or not data:
                    return []
                ids = data[0].split()
                selected = ids[-limit:]
                messages: list[dict[str, Any]] = []
                for message_id in reversed(selected):
                    fetch_status, payload = client.fetch(message_id, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
                    if fetch_status != "OK" or not payload:
                        continue
                    header_bytes = payload[0][1] if isinstance(payload[0], tuple) else b""
                    header_text = header_bytes.decode("utf-8", errors="ignore")
                    lines = [line.strip() for line in header_text.splitlines() if line.strip()]
                    item: dict[str, Any] = {"id": message_id.decode("utf-8", errors="ignore")}
                    for line in lines:
                        if ":" not in line:
                            continue
                        key, value = line.split(":", 1)
                        key_norm = key.strip().lower()
                        item[key_norm] = value.strip()
                    if "date" in item:
                        try:
                            item["date_parsed"] = email.utils.parsedate_to_datetime(item["date"]).isoformat()
                        except Exception:  # noqa: BLE001
                            pass
                    messages.append(item)
                return messages
        except Exception as exc:  # noqa: BLE001
            raise IntegrationServiceError("IMAP list request failed") from exc
