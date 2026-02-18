import httpx


class QdrantHealthClient:
    def __init__(self, qdrant_url: str):
        self._qdrant_url = qdrant_url.rstrip("/")

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._qdrant_url}/collections")
                response.raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False

