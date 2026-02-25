from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blaire_core.config import ensure_runtime_config, read_config_snapshot
from blaire_core.interfaces.cli import _handle_admin, _is_allowed_restricted


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
    runtime_cfg = ensure_runtime_config(snapshot, env="dev")
    assert runtime_cfg.llm.model == "bootstrap-fallback"


def test_local_config_is_merged_before_env_and_cli(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path
    config_dir = repo / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dev.json").write_text(
        json.dumps(
            {
                "app": {"env": "dev"},
                "paths": {"data_root": "./data", "log_dir": "data/logs"},
                "llm": {
                    "base_url": "http://localhost:11434",
                    "model": "base-model",
                    "timeout_seconds": 30,
                    "temperature": 0.75,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 8192,
                },
                "heartbeat": {"interval_seconds": 10},
                "tools": {
                    "web_search": {
                        "api_key": "",
                        "timeout_seconds": 10,
                        "cache_ttl_minutes": 15,
                        "result_count": 10,
                        "safesearch": "off",
                        "auto_use": True,
                        "auto_count": 3,
                    }
                },
                "prompt": {"soul_rules": "base"},
                "session": {
                    "recent_pairs": 6,
                    "maintenance": {
                        "mode": "warn",
                        "prune_after": "30d",
                        "max_entries": 500,
                        "max_disk_bytes": None,
                        "high_water_ratio": 0.8,
                    },
                },
                "logging": {"level": "info"},
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "local.json").write_text(
        json.dumps({"llm": {"model": "local-model"}, "heartbeat": {"interval_seconds": 33}}),
        encoding="utf-8",
    )

    import blaire_core.config as cfg

    monkeypatch.setattr(cfg, "_repo_root", lambda: repo)
    monkeypatch.setenv("BLAIRE_HEARTBEAT_INTERVAL", "44")
    snapshot = read_config_snapshot("dev", {"heartbeat.interval_seconds": "55"})
    assert snapshot.valid
    assert snapshot.effective_config is not None
    assert snapshot.effective_raw is not None
    assert snapshot.effective_raw["llm"]["model"] == "local-model"
    assert snapshot.effective_config.heartbeat.interval_seconds == 55


def test_admin_config_effective_outputs_merged_view(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = tmp_path
    config_dir = repo / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dev.json").write_text(
        json.dumps(
            {
                "app": {"env": "dev"},
                "paths": {"data_root": "./data", "log_dir": "data/logs"},
                "llm": {
                    "base_url": "http://localhost:11434",
                    "model": "model-a",
                    "timeout_seconds": 30,
                    "temperature": 0.75,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 8192,
                },
                "heartbeat": {"interval_seconds": 10},
                "tools": {
                    "web_search": {
                        "api_key": "",
                        "timeout_seconds": 10,
                        "cache_ttl_minutes": 15,
                        "result_count": 10,
                        "safesearch": "off",
                        "auto_use": True,
                        "auto_count": 3,
                    }
                },
                "prompt": {"soul_rules": "base"},
                "session": {
                    "recent_pairs": 6,
                    "maintenance": {
                        "mode": "warn",
                        "prune_after": "30d",
                        "max_entries": 500,
                        "max_disk_bytes": None,
                        "high_water_ratio": 0.8,
                    },
                },
                "logging": {"level": "info"},
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "local.json").write_text(json.dumps({"llm": {"model": "model-local"}}), encoding="utf-8")

    import blaire_core.config as cfg

    monkeypatch.setattr(cfg, "_repo_root", lambda: repo)
    snapshot = read_config_snapshot("dev", {})
    runtime_cfg = ensure_runtime_config(snapshot, env="dev")
    context = SimpleNamespace(config_snapshot=snapshot, config=runtime_cfg)

    _handle_admin(context, ["/admin", "config", "--effective"])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["valid"] is True
    assert payload["effective_source"] == "snapshot_merged"
    assert payload["effective_config"]["llm"]["model"] == "model-local"
