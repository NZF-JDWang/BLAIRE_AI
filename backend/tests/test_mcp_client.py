import pytest

from app.services.mcp_client import McpClient, McpClientError


class FakeResponse:
    def __init__(self, status_code: int, data: dict | None = None):
        self.status_code = status_code
        self._data = data or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict:
        return self._data


@pytest.mark.anyio
async def test_mcp_client_retries_then_succeeds(monkeypatch) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout: float):  # noqa: ARG002
            self.calls = 0

        async def post(self, url: str, json: dict):  # noqa: ANN202
            _ = (url, json)
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary error")
            return FakeResponse(200, {"ok": True})

        async def get(self, url: str):  # noqa: ANN202
            _ = url
            return FakeResponse(200, {"status": "ok"})

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("app.services.mcp_client.httpx.AsyncClient", FakeAsyncClient)
    client = McpClient(retries=2)
    result = await client.call("http://mcp:3000", "vault.read", {"path": "a.md"})
    assert result["ok"] is True
    await client.close()


@pytest.mark.anyio
async def test_mcp_client_raises_after_retry_budget(monkeypatch) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout: float):  # noqa: ARG002
            return None

        async def post(self, url: str, json: dict):  # noqa: ANN202
            _ = (url, json)
            raise RuntimeError("always failing")

        async def get(self, url: str):  # noqa: ANN202
            _ = url
            return FakeResponse(503)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("app.services.mcp_client.httpx.AsyncClient", FakeAsyncClient)
    client = McpClient(retries=1)
    with pytest.raises(McpClientError, match="MCP call failed after retries"):
        await client.call("http://mcp:3000", "vault.read", {"path": "a.md"})
    assert await client.ping("http://mcp:3000") is False
    await client.close()
