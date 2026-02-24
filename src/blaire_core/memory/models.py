"""Memory models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Profile:
    name: str = ""
    environment_summary: str = ""
    long_term_goals: list[str] = field(default_factory=list)
    behavioral_constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Preferences:
    response_style: str = "concise"
    autonomy_level: str = "observe"
    quiet_hours: list[str] = field(default_factory=lambda: ["23:00", "08:00"])
    notification_limits: dict[str, int] = field(default_factory=lambda: {"max_per_day": 5})


@dataclass(slots=True)
class Project:
    id: str
    name: str
    description: str
    status: str
    priority: str
    summary_card: str
    next_actions: list[str]


@dataclass(slots=True)
class Todo:
    id: str
    project_id: str
    title: str
    description: str
    priority: str
    status: str
    created_at: str
    last_updated: str


@dataclass(slots=True)
class Fact:
    id: str
    type: str
    text: str
    tags: list[str]
    importance: float
    created_at: str
    last_used: str | None = None


@dataclass(slots=True)
class SessionMessage:
    role: str
    content: str
    timestamp: str


@dataclass(slots=True)
class SessionRecord:
    id: str
    created_at: str
    messages: list[SessionMessage] = field(default_factory=list)
    running_summary: str = ""


def now_iso_local() -> str:
    """Return machine-local ISO8601 timestamp with offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")

