from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.rag.retrieval import IngestionPipeline
from app.rag.vector_store import QdrantVectorStore, VectorStoreError


@dataclass(frozen=True)
class ObsidianReindexResult:
    scanned_files: int
    indexed_files: int
    unchanged_files: int
    skipped_files: int
    failed_files: int
    chunks_indexed: int
    started_at: datetime


class ObsidianVaultIndexer:
    def __init__(self, vault_path: str):
        self._vault = Path(vault_path)
        self._last_scan_at: datetime | None = None

    def scan_markdown_files(self, limit: int = 5000) -> list[Path]:
        if not self._vault.exists():
            return []
        files: list[Path] = []
        for path in self._vault.rglob("*.md"):
            if not path.is_file():
                continue
            files.append(path)
            if len(files) >= limit:
                break
        self._last_scan_at = datetime.now(timezone.utc)
        return files

    async def reindex(
        self,
        *,
        pipeline: IngestionPipeline,
        vector_store: QdrantVectorStore,
        full_rescan: bool = False,
        limit: int = 5000,
    ) -> ObsidianReindexResult:
        files = self.scan_markdown_files(limit=limit)
        indexed = 0
        unchanged = 0
        failed = 0
        chunks_indexed = 0

        for file_path in files:
            try:
                current_modified = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
                previous_modified = None if full_rescan else await vector_store.get_source_last_modified(file_path)
                if previous_modified and previous_modified == current_modified:
                    unchanged += 1
                    continue
                indexed += 1
                chunks_indexed += await pipeline.ingest_file(file_path)
            except (OSError, VectorStoreError, ValueError):
                failed += 1

        return ObsidianReindexResult(
            scanned_files=len(files),
            indexed_files=indexed,
            unchanged_files=unchanged,
            skipped_files=0,
            failed_files=failed,
            chunks_indexed=chunks_indexed,
            started_at=datetime.now(timezone.utc),
        )

    @property
    def last_scan_at(self) -> datetime | None:
        return self._last_scan_at

