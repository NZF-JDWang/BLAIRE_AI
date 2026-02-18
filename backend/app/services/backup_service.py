from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

from app.models.ops import BackupResponse


class BackupService:
    def __init__(self, backup_root: str, database_url: str, qdrant_url: str):
        self._backup_root = Path(backup_root)
        self._database_url = database_url
        self._qdrant_url = qdrant_url

    def run_backup(self, include_postgres: bool, include_qdrant: bool) -> BackupResponse:
        ts = datetime.now(timezone.utc)
        stamp = ts.strftime("%Y%m%dT%H%M%SZ")
        target_dir = self._backup_root / stamp
        target_dir.mkdir(parents=True, exist_ok=True)
        created: list[str] = []

        if include_postgres:
            pg_file = target_dir / "postgres.sql"
            if self._pg_dump(pg_file):
                created.append(str(pg_file))
            else:
                marker = target_dir / "postgres_not_available.txt"
                marker.write_text("pg_dump unavailable or failed", encoding="utf-8")
                created.append(str(marker))

        if include_qdrant:
            qdrant_meta = target_dir / "qdrant_metadata.json"
            qdrant_meta.write_text(
                json.dumps(
                    {
                        "qdrant_url": self._qdrant_url,
                        "note": "Volume-level snapshot should be handled externally.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            created.append(str(qdrant_meta))

        manifest = target_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "created_at": ts.isoformat(),
                    "include_postgres": include_postgres,
                    "include_qdrant": include_qdrant,
                    "files": created,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        created.append(str(manifest))

        return BackupResponse(backup_dir=str(target_dir), created_at=ts, files=created)

    def _pg_dump(self, out_file: Path) -> bool:
        cmd = ["pg_dump", self._database_url, "-f", str(out_file)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
            return proc.returncode == 0
        except Exception:
            return False

