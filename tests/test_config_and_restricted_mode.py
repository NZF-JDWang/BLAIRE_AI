from __future__ import annotations

import json
from pathlib import Path

from blaire_core.config import read_config_snapshot
from blaire_core.interfaces.cli import _is_allowed_restricted


def test_restricted_mode_allowlist() -> None:
    assert _is_allowed_restricted("/help")
    assert _is_allowed_restricted("/admin diagnostics --deep")
    assert _is_allowed_restricted("/health")
    assert not _is_allowed_restricted("/tool web_search {}")
    assert not _is_allowed_restricted("/session new")


def test_invalid_config_snapshot(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path
    config_dir = repo / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dev.json").write_text(json.dumps({"app": {"env": "dev"}}), encoding="utf-8")

    import blaire_core.config as cfg

    monkeypatch.setattr(cfg, "_repo_root", lambda: repo)
    snapshot = read_config_snapshot("dev", {})
    assert snapshot.exists
    assert not snapshot.valid
    assert snapshot.effective_config is None
    assert snapshot.issues

