import pytest

from app.rag.qdrant_bootstrap import QdrantBootstrapService, collection_payload


def test_collection_payload_shape() -> None:
    payload = collection_payload(768)
    assert payload["vectors"]["size"] == 768
    assert payload["vectors"]["distance"] == "Cosine"


@pytest.mark.anyio
async def test_ensure_collection_creates_on_404(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("http error")

    class FakeClient:
        async def __aenter__(self):  # noqa: ANN204
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
            return False

        async def get(self, url: str):
            calls.append(("GET", url))
            return FakeResponse(404)

        async def put(self, url: str, json: dict):
            calls.append(("PUT", url))
            assert json["vectors"]["size"] == 512
            return FakeResponse(200)

    monkeypatch.setattr("app.rag.qdrant_bootstrap.httpx.AsyncClient", lambda timeout: FakeClient())
    service = QdrantBootstrapService("http://qdrant:6333", "collection_a", 512)
    await service.ensure_collection()
    assert calls[0][0] == "GET"
    assert calls[1][0] == "PUT"

