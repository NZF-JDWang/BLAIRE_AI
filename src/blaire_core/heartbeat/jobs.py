"""Heartbeat job runner."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from blaire_core.config import AppConfig
from blaire_core.memory_store import JsonMemoryStore

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _parse_iso(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Treat naive timestamps as UTC so comparison against aware datetimes
        # remains safe and deterministic.
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ensure_memory_namespace(store: JsonMemoryStore) -> None:
    store.ensure_memory_namespace()
    logger.info("heartbeat: ensured memory namespace")


def _check_due_predictions(store: JsonMemoryStore, now: datetime) -> None:
    predictions = store.load_predictions()
    due_count = 0
    for item in predictions:
        if not isinstance(item, dict):
            continue
        if str(item.get("outcome", "pending")) != "pending":
            continue
        check_after = _parse_iso(str(item.get("check_after", "")))
        if check_after is None:
            continue
        if check_after > now:
            continue
        item["last_checked_at"] = _now_iso()
        due_count += 1
    if due_count:
        store.save_predictions(predictions)
    logger.info("heartbeat: found %s predictions due for evaluation", due_count)


def _update_patterns_heartbeat_marker(store: JsonMemoryStore, now: datetime) -> None:
    patterns = store.load_patterns()
    marker: dict[str, Any] = {
        "id": "heartbeat_last_run",
        "description": "Last heartbeat run",
        "data": {"ok": True},
        "updated_at": now.isoformat(),
    }
    updated = False
    for idx, pattern in enumerate(patterns):
        if isinstance(pattern, dict) and pattern.get("id") == "heartbeat_last_run":
            patterns[idx] = marker
            updated = True
            break
    if not updated:
        patterns.append(marker)
    store.save_patterns(patterns)
    logger.info("heartbeat: updated patterns heartbeat_last_run")


def run_heartbeat_jobs(config: AppConfig) -> None:
    """Run lightweight jobs for one heartbeat tick."""
    now = datetime.now().astimezone()
    store = JsonMemoryStore(config.paths.data_root)
    _ensure_memory_namespace(store)
    _check_due_predictions(store, now)
    _update_patterns_heartbeat_marker(store, now)
