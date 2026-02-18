import asyncio
from typing import Any

import httpx


class McpClientError(RuntimeError):
    pass


class McpClient:
    def __init__(self, timeout_seconds: float = 15.0, retries: int = 2):
        self._timeout = timeout_seconds
        self._retries = retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "McpClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        await self.close()
        return False

    async def connect(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.connect()
        if self._client is None:
            raise McpClientError("Failed to initialize MCP client")
        return self._client

    async def ping(self, base_url: str) -> bool:
        client = await self._ensure_client()
        try:
            response = await client.get(base_url.rstrip("/") + "/health")
            return response.status_code < 500
        except Exception:  # noqa: BLE001
            return False

    async def call(self, base_url: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        url = base_url.rstrip("/") + "/call"
        payload = {"method": method, "params": params}
        last_error: Exception | None = None
        client = await self._ensure_client()
        for attempt in range(self._retries + 1):
            try:
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
