"""Daily episodic compression from structured events into memories/patterns."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from blaire_core.memory_store import StructuredMemoryStore


_NAME_RE = re.compile(r"\bmy name is\s+([^.!?\n]{1,60})", re.IGNORECASE)
_GOAL_RE = re.compile(r"\bmy goal is\s+([^.!?\n]{3,180})", re.IGNORECASE)
_PREF_RE = re.compile(r"\b(i prefer|please be)\s+([^.!?\n]{2,100})", re.IGNORECASE)
_DECISION_RE = re.compile(r"\b(we decided|decision|decided)\b", re.IGNORECASE)
_INCIDENT_RE = re.compile(r"\b(error|failed|failure|incident|outage|vram)\b", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _clean(value: str) -> str:
    return " ".join(value.split()).strip()


def _heuristic_extract(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memories: list[dict[str, Any]] = []
    patterns: list[dict[str, Any]] = []
    event_type_counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        payload = event.get("payload", {})
        text = _clean(str(payload.get("content", "")))
        if not text:
            continue
        name_match = _NAME_RE.search(text)
        if name_match:
            name = _clean(name_match.group(1))
            memories.append(
                {
                    "type": "fact",
                    "text": f"JD's name is {name}.",
                    "importance": 5,
                    "tags": ["profile", "identity", "name"],
                }
            )
        goal_match = _GOAL_RE.search(text)
        if goal_match:
            goal = _clean(goal_match.group(1))
            memories.append(
                {
                    "type": "fact",
                    "text": f"JD long-term goal: {goal}",
                    "importance": 5,
                    "tags": ["profile", "long_term_goal"],
                }
            )
        pref_match = _PREF_RE.search(text)
        if pref_match:
            preference_text = _clean(pref_match.group(2))
            memories.append(
                {
                    "type": "preference",
                    "text": f"JD preference: {preference_text}",
                    "importance": 4,
                    "tags": ["preferences"],
                }
            )
        if _DECISION_RE.search(text):
            memories.append(
                {
                    "type": "decision",
                    "text": text[:512],
                    "importance": 4,
                    "tags": ["decision"],
                }
            )
        if _INCIDENT_RE.search(text):
            memories.append(
                {
                    "type": "incident",
                    "text": text[:512],
                    "importance": 3,
                    "tags": ["incident"],
                }
            )

    for event_type, count in event_type_counts.items():
        if count >= 5:
            patterns.append(
                {
                    "text": f"High frequency pattern: '{event_type}' occurred {count} times in the last 24h.",
                    "importance": 3,
                    "tags": ["system_pattern"],
                }
            )
    return memories, patterns


def _llm_extract(events: list[dict[str, Any]], llm_client: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if llm_client is None:
        return [], []
    sample_rows = []
    for item in events[:200]:
        payload = item.get("payload", {})
        content = _clean(str(payload.get("content", "")))[:200]
        sample_rows.append(f"{item.get('timestamp', '')} | {item.get('event_type', '')} | {content}")
    if not sample_rows:
        return [], []
    prompt = (
        "From this event list, extract new stable facts about JD, decisions made, notable incidents, and possible behaviour "
        'patterns. Return JSON with keys "memories" and "patterns". Each memory item: {"type","text","importance","tags"}. '
        'Each pattern item: {"text","importance","tags"}.'
    )
    response = llm_client.generate(
        system_prompt=prompt,
        messages=[{"role": "user", "content": "\n".join(sample_rows)}],
        max_tokens=900,
    )
    parsed = _extract_json_payload(response)
    if parsed is None:
        return [], []
    raw_memories = parsed.get("memories", []) if isinstance(parsed, dict) else []
    raw_patterns = parsed.get("patterns", []) if isinstance(parsed, dict) else []
    memories = _coerce_items(raw_memories, include_type=True)
    patterns = _coerce_items(raw_patterns, include_type=False)
    return memories, patterns


def _extract_json_payload(response: str) -> dict[str, Any] | None:
    text = response.strip()
    if not text:
        return None
    # Strip common markdown fences first.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_items(raw_items: Any, include_type: bool) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = _clean(str(item.get("text", "")))[:512]
        if not text:
            continue
        importance_raw = item.get("importance", 3)
        try:
            importance = max(1, min(5, int(importance_raw)))
        except Exception:  # noqa: BLE001
            importance = 3
        tags = item.get("tags", [])
        tags_clean = [str(tag) for tag in tags] if isinstance(tags, list) else []
        row: dict[str, Any] = {"text": text, "importance": importance, "tags": tags_clean}
        if include_type:
            row["type"] = str(item.get("type", "fact")).lower()
        out.append(row)
    return out


def should_run_daily_summariser(data_root: str) -> bool:
    store = StructuredMemoryStore(data_root)
    last_run = store.get_meta("summariser.last_run_at")
    if not last_run:
        return True
    text = last_run.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now().astimezone()
    return (now - parsed.astimezone(now.tzinfo)).total_seconds() >= 86400


def run_daily_summariser(data_root: str, llm_client: Any = None) -> dict[str, Any]:
    store = StructuredMemoryStore(data_root)
    store.initialize()
    now = datetime.now().astimezone()
    since = now - timedelta(hours=24)
    events = store.get_events_since(since.isoformat(), limit=2000, unsummarised_only=True)
    if not events:
        store.set_meta("summariser.last_run_at", now.isoformat())
        return {"processed_events": 0, "memories_written": 0, "patterns_written": 0}

    llm_memories, llm_patterns = _llm_extract(events, llm_client=llm_client)
    heur_memories, heur_patterns = _heuristic_extract(events)
    memory_items = [*heur_memories, *llm_memories]
    pattern_items = [*heur_patterns, *llm_patterns]
    memory_upserts: list[dict[str, Any]] = []
    pattern_upserts: list[dict[str, Any]] = []
    source_window = f"{since.date().isoformat()}..{now.date().isoformat()}"
    for item in memory_items:
        text = _clean(str(item.get("text", "")))
        if not text:
            continue
        memory_type = str(item.get("type", "fact")).lower()
        tags = item.get("tags", [])
        importance = int(item.get("importance", 3))
        stability = "stable" if memory_type in {"fact", "decision"} else "evolving"
        memory_upserts.append(
            {
                "memory_type": memory_type,
                "text": text[:512],
                "tags": tags if isinstance(tags, list) else [],
                "importance": importance,
                "stability": stability,
                "now": now.isoformat(),
            }
        )
    for item in pattern_items:
        text = _clean(str(item.get("text", "")))
        if not text:
            continue
        tags = item.get("tags", [])
        importance = int(item.get("importance", 3))
        pattern_upserts.append(
            {
                "text": text[:512],
                "tags": tags if isinstance(tags, list) else [],
                "importance": importance,
                "updated_at": now.isoformat(),
            }
        )

    memories_written = store.add_or_update_memories(memory_upserts) if memory_upserts else 0
    patterns_written = store.add_or_update_patterns(pattern_upserts, source_window=source_window) if pattern_upserts else 0
    event_ids = [int(event["id"]) for event in events]
    store.mark_events_summarised(event_ids, summarised_at=_now_iso())
    store.set_meta("summariser.last_run_at", now.isoformat())
    return {
        "processed_events": len(events),
        "memories_written": memories_written,
        "patterns_written": patterns_written,
    }
