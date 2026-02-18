import httpx


def collection_payload(vector_size: int) -> dict:
    return {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine",
        }
    }


class QdrantBootstrapService:
    def __init__(self, qdrant_url: str, collection_name: str, embedding_dim: int):
        self._qdrant_url = qdrant_url.rstrip("/")
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

    async def ensure_collection(self) -> None:
        get_url = f"{self._qdrant_url}/collections/{self._collection_name}"
        put_url = get_url
        async with httpx.AsyncClient(timeout=10.0) as client:
            get_resp = await client.get(get_url)
            if get_resp.status_code == 200:
                return
            if get_resp.status_code not in (404,):
                get_resp.raise_for_status()
            create_resp = await client.put(put_url, json=collection_payload(self._embedding_dim))
            create_resp.raise_for_status()

