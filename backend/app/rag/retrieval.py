from dataclasses import dataclass
from pathlib import Path

from app.rag.chunking import chunk_text, extract_text
from app.rag.vector_store import QdrantVectorStore
from app.services.ollama_client import OllamaClient


@dataclass(frozen=True)
class RetrievalItem:
    source_path: str
    source_name: str
    file_type: str
    chunk_index: int
    score: float
    text: str
    last_modified: str
    ingested_at: str


class RetrievalService:
    def __init__(
        self,
        *,
        ollama_client: OllamaClient,
        vector_store: QdrantVectorStore,
        embedding_model: str,
    ):
        self._ollama = ollama_client
        self._vector = vector_store
        self._embedding_model = embedding_model

    async def retrieve(self, query: str, limit: int = 5) -> list[RetrievalItem]:
        embedding = await self._ollama.embed(self._embedding_model, query)
        rows = await self._vector.search(embedding, limit=limit)
        return [
            RetrievalItem(
                source_path=row["source_path"],
                source_name=row["source_name"],
                file_type=row["file_type"],
                chunk_index=row["chunk_index"],
                score=row["score"],
                text=row["text"],
                last_modified=row["last_modified"],
                ingested_at=row.get("ingested_at", ""),
            )
            for row in rows
        ]


class IngestionPipeline:
    def __init__(
        self,
        *,
        ollama_client: OllamaClient,
        vector_store: QdrantVectorStore,
        embedding_model: str,
    ):
        self._ollama = ollama_client
        self._vector = vector_store
        self._embedding_model = embedding_model

    async def ingest_file(self, path: Path) -> int:
        text = extract_text(path)
        chunks = chunk_text(text)
        if not chunks:
            return 0
        embeddings = []
        for chunk in chunks:
            embedding = await self._ollama.embed(self._embedding_model, chunk.text)
            embeddings.append(embedding)
        return await self._vector.upsert_chunks(path, chunks, embeddings)
