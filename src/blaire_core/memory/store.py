"""File-based memory store with lock and atomic-write hardening."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .models import SessionMessage, SessionRecord, now_iso_local


@dataclass(slots=True)
class LockHandle:
    target_path: str
    lock_path: str


@dataclass(slots=True)
class LockScanResult:
    scanned: int
    stale_found: int
    removed: int
    errors: list[str]


class FileLockTimeoutError(TimeoutError):
    """Timeout while acquiring a file lock, with structured details."""

    def __init__(self, lock_path: str, timeout_seconds: int, attempts: int) -> None:
        self.error = {
            "code": "lock_timeout",
            "message": f"Timed out waiting for lock: {lock_path}",
            "lock_path": lock_path,
            "timeout_seconds": timeout_seconds,
            "attempts": attempts,
        }
        super().__init__(json.dumps(self.error))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_lock_payload(lock_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def acquire_file_lock(
    target_path: str,
    timeout_seconds: int = 10,
    stale_after_seconds: int = 1800,
) -> LockHandle:
    """Acquire an exclusive file lock with stale lock reclaim."""
    lock_path = Path(f"{target_path}.lock")
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = {"pid": os.getpid(), "created_at": now_iso_local()}
            os.write(fd, json.dumps(payload).encode("utf-8"))
            os.close(fd)
            return LockHandle(target_path=target_path, lock_path=str(lock_path))
        except FileExistsError:
            payload = _read_lock_payload(lock_path)
            now_ts = time.time()
            stale = False
            if payload and isinstance(payload.get("pid"), int):
                pid = int(payload["pid"])
                created_raw = str(payload.get("created_at", ""))
                try:
                    created_ts = time.mktime(time.strptime(created_raw[:19], "%Y-%m-%dT%H:%M:%S"))
                except ValueError:
                    created_ts = 0.0
                age = max(0.0, now_ts - created_ts) if created_ts else stale_after_seconds + 1
                if (not _pid_alive(pid)) or age > stale_after_seconds:
                    stale = True
            else:
                stale = True
            if stale:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if time.monotonic() - started >= timeout_seconds:
                raise FileLockTimeoutError(str(lock_path), timeout_seconds=timeout_seconds, attempts=attempt)
            attempt += 1
            time.sleep(min(1.0, 0.05 * attempt))


def release_file_lock(handle: LockHandle) -> None:
    """Release lock file."""
    Path(handle.lock_path).unlink(missing_ok=True)


def clean_stale_locks(root: str, stale_after_seconds: int = 1800) -> LockScanResult:
    """Clean stale lock files recursively under root."""
    scanned = 0
    stale_found = 0
    removed = 0
    errors: list[str] = []
    now_ts = time.time()
    for lock_file in Path(root).rglob("*.lock"):
        scanned += 1
        payload = _read_lock_payload(lock_file)
        stale = False
        if not payload:
            stale = True
        else:
            pid = payload.get("pid")
            created_raw = str(payload.get("created_at", ""))
            try:
                created_ts = time.mktime(time.strptime(created_raw[:19], "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                created_ts = 0.0
            age = max(0.0, now_ts - created_ts) if created_ts else stale_after_seconds + 1
            if not isinstance(pid, int) or (not _pid_alive(pid)) or age > stale_after_seconds:
                stale = True
        if stale:
            stale_found += 1
            try:
                lock_file.unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                errors.append(f"{lock_file}: {exc}")
    return LockScanResult(scanned=scanned, stale_found=stale_found, removed=removed, errors=errors)


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


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            rows.append(raw)
    return rows


class MemoryStore:
    """Memory store backed by JSON/JSONL/Markdown under data root."""

    def __init__(self, data_root: str) -> None:
        self.data_root = Path(data_root)
        self.sessions_dir = self.data_root / "sessions"
        self.episodic_dir = self.data_root / "episodic"
        self.long_term_dir = self.data_root / "long_term"

    def initialize(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_dir.mkdir(parents=True, exist_ok=True)
        self.long_term_dir.mkdir(parents=True, exist_ok=True)
        defaults = {
            "profile.json": {
                "name": "",
                "environment_summary": "",
                "long_term_goals": [],
                "behavioral_constraints": [],
            },
            "preferences.json": {
                "response_style": "concise",
                "autonomy_level": "observe",
                "quiet_hours": ["23:00", "08:00"],
                "notification_limits": {"max_per_day": 5},
            },
            "projects.json": [],
            "todos.json": [],
        }
        for filename, default in defaults.items():
            path = self.data_root / filename
            if not path.exists():
                _atomic_write_json(path, default)
        for filename in ("facts.jsonl", "lessons.jsonl"):
            path = self.long_term_dir / filename
            if not path.exists():
                path.write_text("", encoding="utf-8")
        today = self.episodic_dir / f"{date.today().isoformat()}.md"
        if not today.exists():
            today.write_text(f"# {date.today().isoformat()}\n\n", encoding="utf-8")

    def load_profile(self) -> dict[str, Any]:
        return _read_json(self.data_root / "profile.json", {})

    def load_preferences(self) -> dict[str, Any]:
        return _read_json(self.data_root / "preferences.json", {})

    def load_projects(self) -> list[dict[str, Any]]:
        return _read_json(self.data_root / "projects.json", [])

    def load_todos(self) -> list[dict[str, Any]]:
        return _read_json(self.data_root / "todos.json", [])

    def save_profile(self, profile: dict[str, Any]) -> None:
        path = self.data_root / "profile.json"
        lock = acquire_file_lock(str(path))
        try:
            _atomic_write_json(path, profile)
        finally:
            release_file_lock(lock)

    def save_preferences(self, preferences: dict[str, Any]) -> None:
        path = self.data_root / "preferences.json"
        lock = acquire_file_lock(str(path))
        try:
            _atomic_write_json(path, preferences)
        finally:
            release_file_lock(lock)

    def load_facts(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.long_term_dir / "facts.jsonl")
        rows.sort(key=lambda item: float(item.get("importance", 0.0)), reverse=True)
        return rows[: max(0, limit)]

    def load_lessons(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.long_term_dir / "lessons.jsonl")
        rows.sort(key=lambda item: float(item.get("importance", 0.0)), reverse=True)
        return rows[: max(0, limit)]

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"session-{session_id}.json"

    def load_or_create_session(self, session_id: str) -> SessionRecord:
        path = self._session_path(session_id)
        if path.exists():
            raw = _read_json(path, {})
            messages = [
                SessionMessage(role=m["role"], content=m["content"], timestamp=m["timestamp"])
                for m in raw.get("messages", [])
                if isinstance(m, dict)
            ]
            return SessionRecord(
                id=raw.get("id", session_id),
                created_at=raw.get("created_at", now_iso_local()),
                messages=messages,
                running_summary=raw.get("running_summary", ""),
            )
        record = SessionRecord(id=session_id, created_at=now_iso_local(), messages=[], running_summary="")
        self.save_session(record)
        return record

    def save_session(self, session: SessionRecord) -> None:
        path = self._session_path(session.id)
        lock = acquire_file_lock(str(path))
        try:
            _atomic_write_json(
                path,
                {
                    "id": session.id,
                    "created_at": session.created_at,
                    "messages": [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in session.messages],
                    "running_summary": session.running_summary,
                },
            )
        finally:
            release_file_lock(lock)

    def append_session_message(self, session_id: str, role: str, content: str) -> SessionRecord:
        session = self.load_or_create_session(session_id)
        session.messages.append(SessionMessage(role=role, content=content, timestamp=now_iso_local()))
        user_turns = len([m for m in session.messages if m.role == "user"])
        assistant_turns = len([m for m in session.messages if m.role == "assistant"])
        latest_topic = content[:80].replace("\n", " ")
        session.running_summary = (
            f"Session has {user_turns} user turns and {assistant_turns} assistant turns; "
            f"latest topic: {latest_topic}"
        )
        self.save_session(session)
        return session

    def append_episodic(self, line: str) -> None:
        today = self.episodic_dir / f"{date.today().isoformat()}.md"
        if not today.exists():
            today.write_text(f"# {date.today().isoformat()}\n\n", encoding="utf-8")
        with today.open("a", encoding="utf-8") as handle:
            handle.write(f"- [{now_iso_local()}] {line}\n")

    def append_fact(self, fact_type: str, text: str, tags: list[str], importance: float) -> dict[str, Any]:
        entry = {
            "id": str(uuid.uuid4()),
            "type": fact_type,
            "text": text,
            "tags": tags,
            "importance": importance,
            "created_at": now_iso_local(),
            "last_used": None,
        }
        path = self.long_term_dir / "facts.jsonl"
        lock = acquire_file_lock(str(path))
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        finally:
            release_file_lock(lock)
        return entry

    def append_lesson(self, text: str, tags: list[str], importance: float) -> dict[str, Any]:
        entry = {
            "id": str(uuid.uuid4()),
            "type": "lesson",
            "text": text,
            "tags": tags,
            "importance": importance,
            "created_at": now_iso_local(),
            "last_used": None,
        }
        path = self.long_term_dir / "lessons.jsonl"
        lock = acquire_file_lock(str(path))
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        finally:
            release_file_lock(lock)
        return entry

    def _parse_duration_days(self, value: str) -> int:
        if value.endswith("d"):
            return int(value[:-1])
        return int(value)

    def preview_session_cleanup(self, mode: str, prune_after: str, max_entries: int, max_disk_bytes: int | None, high_water_ratio: float, active_key: str | None = None) -> dict[str, Any]:
        now = time.time()
        days = self._parse_duration_days(prune_after)
        max_age_seconds = days * 24 * 60 * 60
        entries: list[tuple[Path, float]] = []
        for path in self.sessions_dir.glob("session-*.json"):
            entries.append((path, path.stat().st_mtime))
        entries.sort(key=lambda item: item[1], reverse=True)

        prune_candidates = [p for p, ts in entries if now - ts > max_age_seconds and (active_key is None or active_key not in p.name)]
        cap_candidates = [p for idx, (p, _) in enumerate(entries) if idx >= max_entries and (active_key is None or active_key not in p.name)]

        disk_evictions: list[Path] = []
        if max_disk_bytes is not None:
            total = sum(p.stat().st_size for p, _ in entries)
            target = int(max_disk_bytes * high_water_ratio)
            if total > max_disk_bytes:
                for path, _ in sorted(entries, key=lambda item: item[1]):  # oldest first
                    if active_key and active_key in path.name:
                        continue
                    disk_evictions.append(path)
                    total -= path.stat().st_size
                    if total <= target:
                        break
        actions = {
            "mode": mode,
            "prune_candidates": [str(p) for p in prune_candidates],
            "cap_candidates": [str(p) for p in cap_candidates],
            "disk_budget_evictions": [str(p) for p in disk_evictions],
        }
        return actions

    def enforce_session_cleanup(self, preview: dict[str, Any]) -> dict[str, Any]:
        removed: list[str] = []
        for key in ("prune_candidates", "cap_candidates", "disk_budget_evictions"):
            for path_str in preview.get(key, []):
                path = Path(path_str)
                if path.exists():
                    path.unlink()
                    removed.append(path_str)
        return {"removed": removed, "count": len(removed)}

    def run_session_maintenance(
        self,
        mode: str,
        prune_after: str,
        max_entries: int,
        max_disk_bytes: int | None,
        high_water_ratio: float,
        active_key: str | None = None,
    ) -> dict[str, Any]:
        """Run maintenance according to configured mode."""
        preview = self.preview_session_cleanup(
            mode=mode,
            prune_after=prune_after,
            max_entries=max_entries,
            max_disk_bytes=max_disk_bytes,
            high_water_ratio=high_water_ratio,
            active_key=active_key,
        )
        if mode == "enforce":
            applied = self.enforce_session_cleanup(preview)
            return {"mode": mode, "preview": preview, "applied": applied}
        return {"mode": mode, "preview": preview, "applied": None}
