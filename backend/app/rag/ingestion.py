from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class IngestionResult:
    accepted_files: int
    skipped_files: int
    started_at: datetime


class DropFolderIngestionService:
    def __init__(self, drop_folder: str):
        self._drop_folder = Path(drop_folder)
        self._last_scan_at: datetime | None = None

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

    @property
    def last_scan_at(self) -> datetime | None:
        return self._last_scan_at

