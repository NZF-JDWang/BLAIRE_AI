from __future__ import annotations

import json
from pathlib import Path

import pytest

from blaire_core.memory.store import FileLockTimeoutError, acquire_file_lock, clean_stale_locks, release_file_lock


def test_lock_timeout_when_held(tmp_path: Path) -> None:
    target = str(tmp_path / "x.json")
    handle = acquire_file_lock(target)
    try:
        with pytest.raises(FileLockTimeoutError) as exc:
            acquire_file_lock(target, timeout_seconds=1, stale_after_seconds=9999)
        assert exc.value.error["code"] == "lock_timeout"
        assert str(tmp_path / "x.json.lock") in exc.value.error["lock_path"]
    finally:
        release_file_lock(handle)


def test_stale_lock_reclaimed(tmp_path: Path) -> None:
    target = tmp_path / "y.json"
    lock_path = Path(f"{target}.lock")
    lock_path.write_text(json.dumps({"created_at": "2000-01-01T00:00:00+00:00"}), encoding="utf-8")
    handle = acquire_file_lock(str(target), timeout_seconds=1, stale_after_seconds=1)
    release_file_lock(handle)
    assert not lock_path.exists()


def test_clean_stale_locks(tmp_path: Path) -> None:
    lock_path = tmp_path / "z.json.lock"
    lock_path.write_text("{}", encoding="utf-8")
    result = clean_stale_locks(str(tmp_path), stale_after_seconds=1)
    assert result.scanned >= 1
    assert result.removed >= 1
    assert not lock_path.exists()
