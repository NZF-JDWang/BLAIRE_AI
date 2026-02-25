"""Conservative learning updates from explicit user statements."""

from __future__ import annotations

import re
from typing import Any

from blaire_core.memory.store import MemoryStore


_NAME_RE = re.compile(r"\bmy name is\s+([^.!?\n]{1,60})", re.IGNORECASE)
_GOAL_RE = re.compile(r"\bmy goal is\s+([^.!?\n]{3,180})", re.IGNORECASE)
_PREF_CONCISE_RE = re.compile(r"\b(i prefer|please be)\s+(concise|brief)\b", re.IGNORECASE)
_PREF_DETAILED_RE = re.compile(r"\b(i prefer|please be)\s+(detailed|thorough)\b", re.IGNORECASE)
_SYSTEM_UPGRADE_RE = re.compile(
    r"\b(?:we|i)\s+(?:just\s+)?(?:upgraded|updated|changed|added|enabled)\s+(?:your|blaire(?:'s)?)?\s*([^.!?\n]{3,220})",
    re.IGNORECASE,
)


def _clean_text(value: str) -> str:
    return value.strip().strip(". ")


def apply_learning_updates(memory: MemoryStore, user_message: str, assistant_message: str) -> dict[str, Any]:
    """Update identity/preferences on explicit user statements only."""
    _ = assistant_message
    updates: dict[str, Any] = {
        "profile_updates": {},
        "preferences_updates": {},
        "facts_added": 0,
    }

    profile = memory.load_profile()
    preferences = memory.load_preferences()
    changed_profile = False
    changed_preferences = False

    name_match = _NAME_RE.search(user_message)
    if name_match:
        name = _clean_text(name_match.group(1))
        if name and profile.get("name") != name:
            profile["name"] = name
            updates["profile_updates"]["name"] = name
            changed_profile = True
            memory.append_fact("user_fact", f"User name is {name}.", ["identity", "name"], 0.95)
            memory.add_or_update_memory(
                memory_type="fact",
                text=f"JD's name is {name}.",
                tags=["profile", "identity", "name"],
                importance=5,
                stability="stable",
            )
            updates["facts_added"] += 1

    goal_match = _GOAL_RE.search(user_message)
    if goal_match:
        goal = _clean_text(goal_match.group(1))
        goals = [str(v) for v in profile.get("long_term_goals", []) if isinstance(v, str)]
        if goal and goal not in goals:
            goals.append(goal)
            profile["long_term_goals"] = goals
            updates["profile_updates"]["long_term_goals"] = goals
            changed_profile = True
            memory.append_fact("user_fact", f"User goal: {goal}", ["goal"], 0.85)
            memory.add_or_update_memory(
                memory_type="fact",
                text=f"JD long-term goal: {goal}",
                tags=["profile", "long_term_goal"],
                importance=5,
                stability="stable",
            )
            updates["facts_added"] += 1

    if _PREF_CONCISE_RE.search(user_message):
        if preferences.get("response_style") != "concise":
            preferences["response_style"] = "concise"
            updates["preferences_updates"]["response_style"] = "concise"
            changed_preferences = True
            memory.add_or_update_memory(
                memory_type="preference",
                text="JD prefers concise responses.",
                tags=["preferences", "response_style"],
                importance=4,
                stability="evolving",
            )
    elif _PREF_DETAILED_RE.search(user_message):
        if preferences.get("response_style") != "detailed":
            preferences["response_style"] = "detailed"
            updates["preferences_updates"]["response_style"] = "detailed"
            changed_preferences = True
            memory.add_or_update_memory(
                memory_type="preference",
                text="JD prefers detailed responses.",
                tags=["preferences", "response_style"],
                importance=4,
                stability="evolving",
            )

    system_match = _SYSTEM_UPGRADE_RE.search(user_message)
    if system_match:
        change_text = _clean_text(system_match.group(1))
        if change_text:
            memory.add_or_update_memory(
                memory_type="decision",
                text=f"System update noted: {change_text}.",
                tags=["system", "capability", "upgrade"],
                importance=4,
                stability="stable",
            )
            updates["facts_added"] += 1

    if changed_profile:
        memory.save_profile(profile)
    if changed_preferences:
        memory.save_preferences(preferences)

    return updates
