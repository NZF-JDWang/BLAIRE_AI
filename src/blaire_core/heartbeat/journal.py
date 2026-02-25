"""Heartbeat journal jobs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from blaire_core.memory_store import StructuredMemoryStore


_DEEP_CUT_TOPICS = [
    ("queueing-theory", "Queueing Theory"),
    ("error-budgets", "Error Budgets"),
    ("network-effects", "Network Effects"),
    ("power-law-failures", "Power-Law Failures"),
]


def _today() -> str:
    return datetime.now().astimezone().date().isoformat()


def run_night_log_job(data_root: str) -> str | None:
    date_key = _today()
    root = Path(data_root) / "journal"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{date_key}.md"
    if target.exists():
        return None

    store = StructuredMemoryStore(data_root)
    top_patterns = store.get_top_patterns(limit=5)
    recent_memories = store.get_memories(limit=8)
    recent_events = store.get_events_since(f"{date_key}T00:00:00+00:00", limit=20)

    lines: list[str] = [f"# Internal Night Log {date_key}", "", "## Reflection"]
    lines.append("Today I am compressing activity into durable memory to preserve continuity and reduce drift.")
    lines.append("I am tracking decisions, incidents, and user preferences that should remain available across sessions.")
    lines.append("")
    lines.append("## Patterns")
    if top_patterns:
        for row in top_patterns:
            lines.append(f"- {row.get('text', '')}")
    else:
        lines.append("- No strong patterns were detected today.")
    lines.append("")
    lines.append("## Salient Memories")
    if recent_memories:
        for row in recent_memories[:6]:
            lines.append(f"- [{row.get('type', 'fact')}] {row.get('text', '')}")
    else:
        lines.append("- No salient memories yet.")
    lines.append("")
    lines.append("## Event Trace")
    if recent_events:
        for row in recent_events[-8:]:
            payload = row.get("payload", {})
            content = str(payload.get("content", ""))[:120].replace("\n", " ")
            lines.append(f"- {row.get('event_type', '')}: {content}")
    else:
        lines.append("- No events captured today.")
    lines.append("")
    lines.append(
        "## Next Drift Guard\nI should keep memory retrieval focused on high-importance stable facts while allowing evolving preferences to update."
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def run_deep_cut_job(data_root: str, enabled: bool = False) -> str | None:
    if not enabled:
        return None
    date_key = _today()
    root = Path(data_root) / "journal" / "deep_cuts"
    root.mkdir(parents=True, exist_ok=True)
    index = datetime.now().astimezone().toordinal() % len(_DEEP_CUT_TOPICS)
    slug, title = _DEEP_CUT_TOPICS[index]
    target = root / f"{date_key}_{slug}.md"
    if target.exists():
        return None
    lines = [
        f"# Deep Cut: {title}",
        "",
        "## Claim",
        f"Small structural constraints in {title.lower()} create disproportionate outcomes over time.",
        "",
        "## Math Sketch",
        "Assume throughput T and arrival rate A. When A approaches T, queue delay grows non-linearly.",
        "If utilisation u = A/T and u -> 1, expected wait tends toward a steep increase rather than linear growth.",
        "",
        "## Implication",
        "Systems should reserve slack early; optimising for peak utilisation can quietly increase fragility.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)
