from pathlib import Path

from app.services.backup_service import BackupService


def test_backup_service_writes_manifest(tmp_path: Path) -> None:
    service = BackupService(str(tmp_path), "postgresql://u:p@localhost:5432/db", "http://qdrant:6333")
    result = service.run_backup(include_postgres=False, include_qdrant=True)
    assert Path(result.backup_dir).exists()
    assert any(name.endswith("manifest.json") for name in result.files)

