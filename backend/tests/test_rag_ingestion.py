from pathlib import Path

from app.rag.ingestion import DropFolderIngestionService


def test_scan_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("hello", encoding="utf-8")
    (tmp_path / "doc.pdf").write_text("fake", encoding="utf-8")
    (tmp_path / "script.exe").write_text("bad", encoding="utf-8")

    service = DropFolderIngestionService(str(tmp_path))
    files, skipped = service.scan_files(limit=10)

    file_names = sorted([f.name for f in files])
    assert file_names == ["doc.pdf", "note.md"]
    assert skipped == 1


def test_ingest_returns_counts(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = DropFolderIngestionService(str(tmp_path))
    result = service.ingest(full_rescan=True, limit=10)
    assert result.accepted_files == 1
    assert result.skipped_files == 0
    assert result.started_at is not None

