from __future__ import annotations

import json
from pathlib import Path

from blaire_core.config import (
    AppConfig,
    AppSection,
    HeartbeatSection,
    LLMSection,
    LoggingSection,
    PathsSection,
    PromptSection,
    SessionMaintenanceSection,
    SessionSection,
    ToolsSection,
    WebSearchSection,
)
from blaire_core.heartbeat.jobs import run_heartbeat_jobs


def _config_for_path(tmp_path: Path) -> AppConfig:
    return AppConfig(
        app=AppSection(env="dev"),
        paths=PathsSection(data_root=str(tmp_path), log_dir=str(tmp_path / "logs")),
        llm=LLMSection(base_url="http://localhost", model="stub", timeout_seconds=30),
        heartbeat=HeartbeatSection(interval_seconds=0),
        tools=ToolsSection(
            web_search=WebSearchSection(
                api_key="",
                timeout_seconds=10,
                cache_ttl_minutes=15,
                result_count=10,
                safesearch="off",
                auto_use=True,
                auto_count=3,
            )
        ),
        prompt=PromptSection(soul_rules="Be useful."),
        session=SessionSection(
            recent_pairs=6,
            maintenance=SessionMaintenanceSection(
                mode="warn",
                prune_after="30d",
                max_entries=500,
                max_disk_bytes=None,
                high_water_ratio=0.8,
            ),
        ),
        logging=LoggingSection(level="info"),
    )


def test_heartbeat_jobs_create_memory_files_and_update_patterns(tmp_path: Path) -> None:
    config = _config_for_path(tmp_path)

    run_heartbeat_jobs(config)

    memory_dir = tmp_path / "memory"
    assert (memory_dir / "projects.json").exists()
    assert (memory_dir / "todo.json").exists()
    assert (memory_dir / "debates.json").exists()
    assert (memory_dir / "predictions.json").exists()
    patterns = json.loads((memory_dir / "patterns.json").read_text(encoding="utf-8"))
    assert any(p.get("id") == "heartbeat_last_run" for p in patterns)


def test_heartbeat_jobs_marks_due_predictions(tmp_path: Path) -> None:
    config = _config_for_path(tmp_path)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "projects.json").write_text("[]", encoding="utf-8")
    (memory_dir / "todo.json").write_text("[]", encoding="utf-8")
    (memory_dir / "debates.json").write_text("[]", encoding="utf-8")
    (memory_dir / "patterns.json").write_text("[]", encoding="utf-8")
    (memory_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "id": "pred-due",
                    "statement": "something",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "check_after": "2000-01-01T00:00:00+00:00",
                    "outcome": "pending",
                    "notes": None,
                    "related_debate_id": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    run_heartbeat_jobs(config)

    predictions = json.loads((memory_dir / "predictions.json").read_text(encoding="utf-8"))
    assert predictions[0]["id"] == "pred-due"
    assert "last_checked_at" in predictions[0]


def test_heartbeat_jobs_handles_naive_check_after_timestamps(tmp_path: Path) -> None:
    config = _config_for_path(tmp_path)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "projects.json").write_text("[]", encoding="utf-8")
    (memory_dir / "todo.json").write_text("[]", encoding="utf-8")
    (memory_dir / "debates.json").write_text("[]", encoding="utf-8")
    (memory_dir / "patterns.json").write_text("[]", encoding="utf-8")
    (memory_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "id": "pred-naive",
                    "statement": "something else",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "check_after": "2000-01-01T00:00:00",
                    "outcome": "pending",
                    "notes": None,
                    "related_debate_id": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    run_heartbeat_jobs(config)

    predictions = json.loads((memory_dir / "predictions.json").read_text(encoding="utf-8"))
    assert predictions[0]["id"] == "pred-naive"
    assert "last_checked_at" in predictions[0]


def test_heartbeat_jobs_skips_future_naive_check_after_timestamps(tmp_path: Path) -> None:
    config = _config_for_path(tmp_path)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "projects.json").write_text("[]", encoding="utf-8")
    (memory_dir / "todo.json").write_text("[]", encoding="utf-8")
    (memory_dir / "debates.json").write_text("[]", encoding="utf-8")
    (memory_dir / "patterns.json").write_text("[]", encoding="utf-8")
    (memory_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "id": "pred-naive-future",
                    "statement": "future check",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "check_after": "2999-01-01T00:00:00",
                    "outcome": "pending",
                    "notes": None,
                    "related_debate_id": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    run_heartbeat_jobs(config)

    predictions = json.loads((memory_dir / "predictions.json").read_text(encoding="utf-8"))
    assert predictions[0]["id"] == "pred-naive-future"
    assert "last_checked_at" not in predictions[0]
