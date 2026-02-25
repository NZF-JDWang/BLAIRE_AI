from __future__ import annotations

import json
from pathlib import Path

from blaire_core.memory_store import JsonMemoryStore


def test_memory_store_uses_env_data_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLAIRE_DATA_PATH", str(tmp_path))
    store = JsonMemoryStore()
    store.ensure_memory_namespace()

    assert store.memory_dir == tmp_path / "memory"
    assert (tmp_path / "memory" / "projects.json").exists()
    assert (tmp_path / "memory" / "todo.json").exists()
    assert (tmp_path / "memory" / "debates.json").exists()
    assert (tmp_path / "memory" / "predictions.json").exists()
    assert (tmp_path / "memory" / "patterns.json").exists()


def test_memory_store_recovers_corrupt_file(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path)
    store.ensure_memory_namespace()
    corrupt_path = tmp_path / "memory" / "projects.json"
    corrupt_path.write_text("{not-json", encoding="utf-8")

    projects = store.load_projects()

    assert projects == []
    assert corrupt_path.exists()
    assert (tmp_path / "memory" / "projects.json.bak").exists()


def test_memory_store_round_trip_predictions(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path)
    payload = [{"id": "pred-1", "statement": "x", "outcome": "pending"}]

    store.save_predictions(payload)

    assert store.load_predictions() == payload
    raw = json.loads((tmp_path / "memory" / "predictions.json").read_text(encoding="utf-8"))
    assert raw == payload
