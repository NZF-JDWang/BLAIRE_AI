"""JSON-backed assistant memory namespace helpers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_MEMORY_FILES: dict[str, Any] = {
    "projects.json": [],
    "todo.json": [],
    "debates.json": [],
    "predictions.json": [],
    "patterns.json": [],
}


class JsonMemoryStore:
    """Helpers for reading and writing memory JSON files under data_root/memory."""

    def __init__(self, data_root: str | Path | None = None) -> None:
        self.data_root = self.resolve_data_root(data_root)
        self.memory_dir = self.data_root / "memory"

    @staticmethod
    def resolve_data_root(data_root: str | Path | None = None) -> Path:
        if data_root is not None:
            return Path(data_root)
        env_path = os.getenv("BLAIRE_DATA_PATH")
        if env_path:
            return Path(env_path)
        return Path("./data")

    def ensure_memory_namespace(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for filename, default in _MEMORY_FILES.items():
            path = self.memory_dir / filename
            if not path.exists():
                self._atomic_write_json(path, default)

    def _load_json(self, filename: str, default: Any) -> Any:
        self.ensure_memory_namespace()
        path = self.memory_dir / filename
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                path.replace(backup_path)
            except OSError:
                logger.exception("memory_store: failed to backup corrupt file", extra={"path": str(path)})
            logger.error("memory_store: corrupt file recreated", extra={"path": str(path), "error": str(exc)})
            self._atomic_write_json(path, default)
            return default

    def _save_json(self, filename: str, payload: Any) -> None:
        self.ensure_memory_namespace()
        self._atomic_write_json(self.memory_dir / filename, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load_projects(self) -> list[dict[str, Any]]:
        return self._load_json("projects.json", [])

    def save_projects(self, projects: list[dict[str, Any]]) -> None:
        self._save_json("projects.json", projects)

    def load_todos(self) -> list[dict[str, Any]]:
        return self._load_json("todo.json", [])

    def save_todos(self, todos: list[dict[str, Any]]) -> None:
        self._save_json("todo.json", todos)

    def load_debates(self) -> list[dict[str, Any]]:
        return self._load_json("debates.json", [])

    def save_debates(self, debates: list[dict[str, Any]]) -> None:
        self._save_json("debates.json", debates)

    def load_predictions(self) -> list[dict[str, Any]]:
        return self._load_json("predictions.json", [])

    def save_predictions(self, predictions: list[dict[str, Any]]) -> None:
        self._save_json("predictions.json", predictions)

    def load_patterns(self) -> list[dict[str, Any]]:
        data = self._load_json("patterns.json", [])
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []

    def save_patterns(self, patterns: list[dict[str, Any]]) -> None:
        self._save_json("patterns.json", patterns)
