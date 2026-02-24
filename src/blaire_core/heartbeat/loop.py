"""Heartbeat loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class HeartbeatStatus:
    running: bool
    interval_seconds: int


class HeartbeatLoop:
    """Run heartbeat ticks in background thread."""

    def __init__(self, interval_seconds: int, tick_fn: Callable[[], None]) -> None:
        self._interval_seconds = interval_seconds
        self._tick_fn = tick_fn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def tick_once(self) -> None:
        self._tick_fn()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._interval_seconds <= 0:
            return
        self._stop.clear()

        def _runner() -> None:
            while not self._stop.is_set():
                self._tick_fn()
                self._stop.wait(self._interval_seconds)

        self._thread = threading.Thread(target=_runner, name="blaire-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def status(self) -> HeartbeatStatus:
        return HeartbeatStatus(
            running=bool(self._thread and self._thread.is_alive()),
            interval_seconds=self._interval_seconds,
        )

