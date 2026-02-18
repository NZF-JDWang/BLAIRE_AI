import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.rag.chunking import TextChunk


class VectorStoreError(RuntimeError):
    pass


def chunk_point_id(path: Path, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{path.as_posix()}::{chunk_index}".encode("utf-8")).hexdigest()
    return digest


class QdrantVectorStore:
    def __init__(self, qdrant_url: str, collection_name: str):
        self._base = qdrant_url.rstrip("/")
        self._collection = collection_name

    async def upsert_chunks(
        self,
        path: Path,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunk and embedding lengths differ")

        last_modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                {
                    "id": chunk_point_id(path, chunk.chunk_index),
                    "vector": embedding,
                    "payload": {
                        "source_path": str(path),
                        "source_name": path.name,
                        "file_type": path.suffix.lower(),
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "last_modified": last_modified,
                    },
                }
            )

        url = f"{self._base}/collections/{self._collection}/points?wait=true"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.put(url, json={"points": points})
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("Qdrant upsert failed") from exc
        return len(points)

    async def search(self, vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        url = f"{self._base}/collections/{self._collection}/points/search"
        payload = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("Qdrant search failed") from exc

        results = data.get("result", [])
        normalized: list[dict[str, Any]] = []
        for item in results:
            payload = item.get("payload", {}) or {}
            normalized.append(
                {
                    "score": float(item.get("score", 0.0)),
                    "source_path": str(payload.get("source_path", "")),
                    "source_name": str(payload.get("source_name", "")),
                    "file_type": str(payload.get("file_type", "")),
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "text": str(payload.get("text", "")),
                    "last_modified": str(payload.get("last_modified", "")),
                }
            )
        return normalized

