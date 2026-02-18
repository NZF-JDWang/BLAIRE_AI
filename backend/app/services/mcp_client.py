import asyncio
from typing import Any

import httpx


class McpClientError(RuntimeError):
    pass


class McpClient:
    def __init__(self, timeout_seconds: float = 15.0, retries: int = 2):
        self._timeout = timeout_seconds
        self._retries = retries

    async def call(self, base_url: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        url = base_url.rstrip("/") + "/call"
        payload = {"method": method, "params": params}
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict):
                        raise McpClientError("Invalid MCP response shape")
                    return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self._retries:
                    await asyncio.sleep(0.2 * (attempt + 1))
        raise McpClientError("MCP call failed after retries") from last_error

