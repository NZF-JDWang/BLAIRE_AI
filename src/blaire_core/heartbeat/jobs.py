"""Heartbeat job runner."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from blaire_core.config import AppConfig
from blaire_core.heartbeat.journal import run_deep_cut_job, run_night_log_job
from blaire_core.heartbeat.summarise_events import run_daily_summariser, should_run_daily_summariser
from blaire_core.memory_store import JsonMemoryStore, StructuredMemoryStore

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


def run_heartbeat_jobs(config: AppConfig, llm_client: Any | None = None) -> None:
    """Run lightweight jobs for one heartbeat tick."""
    now = datetime.now().astimezone()
    store = JsonMemoryStore(config.paths.data_root)
    _ensure_memory_namespace(store)
    _check_due_predictions(store, now)
    _update_patterns_heartbeat_marker(store, now)
    structured = StructuredMemoryStore(config.paths.data_root)
    try:
        keep_days = int(os.getenv("BLAIRE_EVENT_RETENTION_DAYS", "30"))
    except ValueError:
        keep_days = 30
    try:
        deleted = structured.prune_old_events(keep_days=max(1, keep_days))
        if deleted:
            logger.info("heartbeat: pruned old events", extra={"deleted": deleted, "keep_days": keep_days})
    except Exception as exc:  # noqa: BLE001
        logger.warning("heartbeat: event retention pruning failed: %s", exc)
    try:
        if should_run_daily_summariser(config.paths.data_root):
            summary = run_daily_summariser(config.paths.data_root, llm_client=llm_client)
            logger.info("heartbeat: daily summariser ran", extra=summary)
        else:
            logger.debug("heartbeat: daily summariser skipped (not due)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("heartbeat: daily summariser failed: %s", exc)
    try:
        path = run_night_log_job(config.paths.data_root)
        if path:
            logger.info("heartbeat: night log written", extra={"path": path})
    except Exception as exc:  # noqa: BLE001
        logger.warning("heartbeat: night log failed: %s", exc)
    deep_cut_enabled = os.getenv("BLAIRE_DEEP_CUT_ENABLED", "").strip().lower() == "true"
    try:
        path = run_deep_cut_job(config.paths.data_root, enabled=deep_cut_enabled)
        if path:
            logger.info("heartbeat: deep cut written", extra={"path": path})
    except Exception as exc:  # noqa: BLE001
        logger.warning("heartbeat: deep cut failed: %s", exc)
