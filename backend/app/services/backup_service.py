from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

import httpx
import psycopg

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

        qdrant_hook = target_dir / "qdrant_hook.json"
        qdrant_hook.write_text(json.dumps(self._collect_qdrant_hook(), indent=2), encoding="utf-8")
        created.append(str(qdrant_hook))

        app_state_hook = target_dir / "app_state_hook.json"
        app_state_hook.write_text(json.dumps(self._collect_app_state_hook(), indent=2), encoding="utf-8")
        created.append(str(app_state_hook))

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

    def _collect_qdrant_hook(self) -> dict:
        url = self._qdrant_url.rstrip("/") + "/collections"
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
            return {"status": "ok", "collections": data.get("result", {}).get("collections", [])}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc), "collections": []}

    def _collect_app_state_hook(self) -> dict:
        db_url = self._database_url.replace("postgresql+psycopg://", "postgresql://")
        tables = ["approvals", "approval_audit_events", "user_preferences", "ingestion_file_state"]
        counts: dict[str, int | None] = {}
        errors: dict[str, str] = {}
        try:
            with psycopg.connect(db_url, autocommit=True, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    for table in tables:
                        try:
                            cur.execute(f"SELECT COUNT(*) FROM {table};")
                            row = cur.fetchone()
                            counts[table] = int(row[0]) if row else 0
                        except Exception as exc:  # noqa: BLE001
                            counts[table] = None
                            errors[table] = str(exc)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc), "table_counts": counts, "table_errors": errors}

        return {"status": "ok", "table_counts": counts, "table_errors": errors}
