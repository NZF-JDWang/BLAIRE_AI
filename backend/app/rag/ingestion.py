from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.rag.retrieval import IngestionPipeline

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class IngestionResult:
    accepted_files: int
    skipped_files: int
    started_at: datetime
    chunks_indexed: int = 0


@dataclass
class WatchIngestionResult:
    scanned_files: int
    indexed_files: int
    skipped_files: int
    failed_files: int
    chunks_indexed: int
    started_at: datetime


@dataclass
class _FileWatchState:
    last_success_mtime: float | None = None
    last_attempt_ts: float = 0.0
    failure_count: int = 0
    next_retry_ts: float = 0.0


class DropFolderIngestionService:
    def __init__(self, drop_folder: str):
        self._drop_folder = Path(drop_folder)
        self._last_scan_at: datetime | None = None
        self._watch_state: dict[str, _FileWatchState] = {}

    def scan_files(self, limit: int = 100) -> tuple[list[Path], int]:
        if not self._drop_folder.exists():
            return [], 0

        accepted: list[Path] = []
        skipped = 0
        for file_path in self._drop_folder.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue
            accepted.append(file_path)
            if len(accepted) >= limit:
                break
        self._last_scan_at = datetime.now(timezone.utc)
        return accepted, skipped

    def ingest(self, full_rescan: bool = False, limit: int = 100) -> IngestionResult:
        _ = full_rescan  # Reserved for delta/full strategy in later implementation.
        accepted, skipped = self.scan_files(limit=limit)
        return IngestionResult(
            accepted_files=len(accepted),
            skipped_files=skipped,
            started_at=datetime.now(timezone.utc),
        )

    async def ingest_with_pipeline(
        self,
        *,
        pipeline: IngestionPipeline,
        full_rescan: bool = False,
        limit: int = 100,
    ) -> IngestionResult:
        _ = full_rescan  # Reserved for delta/full strategy in later implementation.
        files, skipped = self.scan_files(limit=limit)
        indexed_chunks = 0
        for file_path in files:
            indexed_chunks += await pipeline.ingest_file(file_path)
        return IngestionResult(
            accepted_files=len(files),
            skipped_files=skipped,
            started_at=datetime.now(timezone.utc),
            chunks_indexed=indexed_chunks,
        )

    async def ingest_changed_with_retry(
        self,
        *,
        pipeline: IngestionPipeline,
        limit: int = 100,
        debounce_seconds: int = 10,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 300,
        current_time_ts: float | None = None,
    ) -> WatchIngestionResult:
        files, skipped = self.scan_files(limit=limit)
        started_at = datetime.now(timezone.utc)
        now_ts = current_time_ts if current_time_ts is not None else started_at.timestamp()
        indexed_files = 0
        failed_files = 0
        chunks_indexed = 0

        for file_path in files:
            key = str(file_path)
            state = self._watch_state.setdefault(key, _FileWatchState())
            current_mtime = file_path.stat().st_mtime

            unchanged = state.last_success_mtime is not None and current_mtime <= state.last_success_mtime
            if unchanged and state.failure_count == 0:
                skipped += 1
                continue
            if now_ts - state.last_attempt_ts < debounce_seconds:
                skipped += 1
                continue
            if state.next_retry_ts and now_ts < state.next_retry_ts:
                skipped += 1
                continue

            state.last_attempt_ts = now_ts
            try:
                chunks_indexed += await pipeline.ingest_file(file_path)
                indexed_files += 1
                state.last_success_mtime = current_mtime
                state.failure_count = 0
                state.next_retry_ts = 0.0
            except Exception:  # noqa: BLE001
                failed_files += 1
                state.failure_count += 1
                backoff = min(retry_base_seconds * (2 ** (state.failure_count - 1)), retry_max_seconds)
                state.next_retry_ts = now_ts + backoff

        return WatchIngestionResult(
            scanned_files=len(files),
            indexed_files=indexed_files,
            skipped_files=skipped,
            failed_files=failed_files,
            chunks_indexed=chunks_indexed,
            started_at=started_at,
        )

    @property
    def last_scan_at(self) -> datetime | None:
        return self._last_scan_at
