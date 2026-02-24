"""Evolving soul growth updates."""

from __future__ import annotations

import re
from typing import Any

from blaire_core.memory.store import MemoryStore


_MAX_GROWTH_NOTES = 20
_MAX_ALIGNMENT_NOTES = 20
_MAX_NOTE_LEN = 240

_POSITIVE_RE = re.compile(r"\b(that helped|this helped|i like when|keep doing|that works)\b", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"\b(that did not help|that didn't help|i do not like when|i don't like when|stop doing)\b", re.IGNORECASE)


def _normalize_note(value: str) -> str:
    return " ".join(value.split()).strip()[:_MAX_NOTE_LEN]


def _append_unique_bounded(target: list[str], note: str, max_items: int) -> bool:
    normalized = _normalize_note(note)
    if len(normalized) < 12:
        return False
    lowered = normalized.lower()
    if any(str(v).strip().lower() == lowered for v in target):
        return False
    target.append(normalized)
    if len(target) > max_items:
        del target[0 : len(target) - max_items]
    return True


def apply_soul_growth_updates(memory: MemoryStore, user_message: str, assistant_message: str) -> dict[str, Any]:
    """Update evolving soul notes from explicit feedback signals."""
    soul = memory.load_evolving_soul()
    growth_notes = [str(v) for v in soul.get("growth_notes", []) if isinstance(v, str)]
    align_notes = [str(v) for v in soul.get("user_alignment_notes", []) if isinstance(v, str)]

    changed = False
    updates: dict[str, Any] = {
        "growth_notes_added": 0,
        "alignment_notes_added": 0,
    }

    if _POSITIVE_RE.search(user_message):
        if _append_unique_bounded(
            align_notes,
            f"Positive user feedback observed: '{_normalize_note(user_message)}'",
            _MAX_ALIGNMENT_NOTES,
        ):
            updates["alignment_notes_added"] += 1
            changed = True

    if _NEGATIVE_RE.search(user_message):
        if _append_unique_bounded(
            align_notes,
            f"Negative user feedback observed: '{_normalize_note(user_message)}'",
            _MAX_ALIGNMENT_NOTES,
        ):
            updates["alignment_notes_added"] += 1
            changed = True

    if _append_unique_bounded(
        growth_notes,
        f"Recent interaction pattern: user='{_normalize_note(user_message)}' assistant='{_normalize_note(assistant_message)}'",
        _MAX_GROWTH_NOTES,
    ):
        updates["growth_notes_added"] += 1
        changed = True

    if changed:
        soul["growth_notes"] = growth_notes
        soul["user_alignment_notes"] = align_notes
        memory.save_evolving_soul(soul)

    updates["updated"] = changed
    return updates
