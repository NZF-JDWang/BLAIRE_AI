from pathlib import Path

import pytest

from app.rag.obsidian_indexer import ObsidianVaultIndexer


class FakePipeline:
    async def ingest_file(self, path: Path):  # noqa: ANN001, ANN202
        return 2 if path.name.endswith(".md") else 0


class FakeVectorStore:
    def __init__(self):
        self._map: dict[str, str] = {}

    async def get_source_last_modified(self, path: Path):  # noqa: ANN001, ANN202
        return self._map.get(str(path))


@pytest.mark.anyio
async def test_obsidian_reindex_detects_unchanged(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("hello", encoding="utf-8")

    indexer = ObsidianVaultIndexer(str(vault))
    vector = FakeVectorStore()
    current_mtime = __import__("datetime").datetime.fromtimestamp(note.stat().st_mtime, tz=__import__("datetime").timezone.utc).isoformat()
    vector._map[str(note)] = current_mtime

    result = await indexer.reindex(
        pipeline=FakePipeline(),
        vector_store=vector,  # type: ignore[arg-type]
        full_rescan=False,
        limit=100,
    )
    assert result.scanned_files == 1
    assert result.unchanged_files == 1
    assert result.indexed_files == 0


@pytest.mark.anyio
async def test_obsidian_reindex_full_rescan_indexes(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("hello", encoding="utf-8")

    indexer = ObsidianVaultIndexer(str(vault))
    result = await indexer.reindex(
        pipeline=FakePipeline(),
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        full_rescan=True,
        limit=100,
    )
    assert result.indexed_files == 1
    assert result.chunks_indexed == 2

