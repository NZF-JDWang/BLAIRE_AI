"""Outbound notification stub for heartbeat jobs."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from blaire_core.memory_store import JsonMemoryStore

logger = logging.getLogger(__name__)


def notify_user(message: str, *, level: str = "info") -> None:
    """Append a notification entry to data_root/outbox.log and logger."""
    data_root = JsonMemoryStore.resolve_data_root()
    outbox = Path(data_root) / "outbox.log"
    outbox.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat()
    line = f"{now}\t{level}\t{message}\n"
    with outbox.open("a", encoding="utf-8") as handle:
        handle.write(line)
    logger.info("notify_user: %s", line.strip())
