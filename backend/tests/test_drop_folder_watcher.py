import os
from pathlib import Path

import pytest

from app.rag.ingestion import DropFolderIngestionService


class FakePipeline:
    def __init__(self, *, fail_first: bool = False):
        self.calls = 0
        self._fail_first = fail_first

    async def ingest_file(self, path: Path) -> int:  # noqa: ANN201
        _ = path
        self.calls += 1
        if self._fail_first and self.calls == 1:
            raise RuntimeError("temporary failure")
        return 2


def _touch(path: Path, ts: float) -> None:
    os.utime(path, (ts, ts))


@pytest.mark.anyio
async def test_watcher_skips_unchanged_files(tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("hello", encoding="utf-8")
    _touch(doc, 100.0)

    service = DropFolderIngestionService(str(tmp_path))
    pipeline = FakePipeline()

    first = await service.ingest_changed_with_retry(
        pipeline=pipeline,
        debounce_seconds=1,
        current_time_ts=1000.0,
    )
    second = await service.ingest_changed_with_retry(
        pipeline=pipeline,
        debounce_seconds=1,
        current_time_ts=1010.0,
    )

    assert first.indexed_files == 1
    assert second.indexed_files == 0
    assert second.skipped_files >= 1
    assert pipeline.calls == 1


@pytest.mark.anyio
async def test_watcher_applies_debounce_for_changed_file(tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("hello", encoding="utf-8")
    _touch(doc, 100.0)

    service = DropFolderIngestionService(str(tmp_path))
    pipeline = FakePipeline()

    await service.ingest_changed_with_retry(
        pipeline=pipeline,
        debounce_seconds=10,
        current_time_ts=1000.0,
    )

    doc.write_text("hello again", encoding="utf-8")
    _touch(doc, 200.0)

    debounced = await service.ingest_changed_with_retry(
        pipeline=pipeline,
        debounce_seconds=10,
        current_time_ts=1005.0,
    )
    after_debounce = await service.ingest_changed_with_retry(
        pipeline=pipeline,
        debounce_seconds=10,
        current_time_ts=1011.0,
    )

    assert debounced.indexed_files == 0
    assert after_debounce.indexed_files == 1
    assert pipeline.calls == 2


@pytest.mark.anyio
async def test_watcher_retries_with_backoff(tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("hello", encoding="utf-8")
    _touch(doc, 100.0)

    service = DropFolderIngestionService(str(tmp_path))
    pipeline = FakePipeline(fail_first=True)

    first = await service.ingest_changed_with_retry(
        pipeline=pipeline,
        retry_base_seconds=5,
        debounce_seconds=0,
        current_time_ts=2000.0,
    )
    before_retry = await service.ingest_changed_with_retry(
        pipeline=pipeline,
        retry_base_seconds=5,
        debounce_seconds=0,
        current_time_ts=2002.0,
    )
    retried = await service.ingest_changed_with_retry(
        pipeline=pipeline,
        retry_base_seconds=5,
        debounce_seconds=0,
        current_time_ts=2006.0,
    )

    assert first.failed_files == 1
    assert before_retry.indexed_files == 0
    assert retried.indexed_files == 1
    assert pipeline.calls == 2
