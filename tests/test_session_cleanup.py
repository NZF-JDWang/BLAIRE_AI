from __future__ import annotations

import time
from pathlib import Path

from blaire_core.memory.store import MemoryStore


def _touch(path: Path, age_seconds: int) -> None:
    path.write_text("{}", encoding="utf-8")
    ts = time.time() - age_seconds
    path.chmod(0o644)
    path.touch()
    import os

    os.utime(path, (ts, ts))


def test_session_cleanup_dry_run_and_enforce(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    recent = store.sessions_dir / "session-recent.json"
    stale = store.sessions_dir / "session-stale.json"
    _touch(recent, 5)
    _touch(stale, 40 * 24 * 60 * 60)

    preview = store.preview_session_cleanup(
        mode="warn",
        prune_after="30d",
        max_entries=500,
        max_disk_bytes=None,
        high_water_ratio=0.8,
    )
    assert any("session-stale" in p for p in preview["prune_candidates"])
    assert recent.exists()
    assert stale.exists()

    result = store.enforce_session_cleanup(preview)
    assert result["count"] >= 1
    assert recent.exists()
    assert not stale.exists()

