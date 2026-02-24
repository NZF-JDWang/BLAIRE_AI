"""Prompt composer using file templates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from blaire_core.memory.store import MemoryStore


logger = logging.getLogger(__name__)


_TEMPLATE_FILES = {
    "soul": "soul_rules.md",
    "evolving_soul": "evolving_soul.md",
    "core_persona": "soul_core_persona.md",
    "intelligence_contract": "persona_intelligence_contract.md",
    "anti_chatbot_contract": "anti_chatbot_contract.md",
    "living_persona": "soul_living_persona.md",
    "identity": "identity_card.md",
    "prefs": "user_preferences_card.md",
    "projects": "project_cards.md",
    "todos": "todo_cards.md",
    "facts": "long_term_snippets.md",
}


def _template_root() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "reference" / "templates"


def _read_template(name: str) -> str:
    path = _template_root() / _TEMPLATE_FILES[name]
    if not path.exists():
        logger.warning("prompt template missing: %s", path)
        return ""
    return path.read_text(encoding="utf-8")


def _render(template_text: str, values: dict[str, str]) -> str:
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered.strip()


def _list_to_bullets(values: list[str]) -> str:
    if not values:
        return "- (none)"
    return "\n".join(f"- {v}" for v in values)


def _project_cards(projects: list[dict[str, Any]]) -> str:
    if not projects:
        return "- (none)"
    ordered = sorted(
        projects,
        key=lambda p: (str(p.get("status", "")) != "active", str(p.get("priority", "") != "high")),
    )[:2]
    rows: list[str] = []
    for project in ordered:
        rows.append(
            "\n".join(
                [
                    f"- [{project.get('status', 'unknown')}] {project.get('name', '(unnamed)')}",
                    f"  id: {project.get('id', '')}",
                    f"  priority: {project.get('priority', '')}",
                    f"  summary: {project.get('summary_card', project.get('description', ''))}",
                ]
            )
        )
    return "\n".join(rows)


def _todo_cards(todos: list[dict[str, Any]]) -> str:
    if not todos:
        return "- (none)"
    filtered = [t for t in todos if str(t.get("status", "open")) != "done"]
    ranked = sorted(
        filtered,
        key=lambda t: (str(t.get("priority", "later")) != "now", str(t.get("status", "open")) != "in_progress"),
    )[:5]
    if not ranked:
        return "- (none)"
    return "\n".join(f"- [{t.get('priority', 'later')}] {t.get('title', '')}" for t in ranked)


def _fact_snippets(memory: MemoryStore) -> str:
    facts = memory.load_facts(limit=10)
    lessons = memory.load_lessons(limit=5)
    rows: list[str] = []
    for entry in facts + lessons:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        entry_type = str(entry.get("type", "fact"))
        rows.append(f"- ({entry_type}) {text}")
    if not rows:
        return "- (none)"
    return "\n".join(rows[:10])


def _evolving_soul_card(memory: MemoryStore) -> str:
    soul = memory.load_evolving_soul()
    traits = [str(v) for v in soul.get("traits", []) if isinstance(v, str)]
    align = [str(v) for v in soul.get("user_alignment_notes", []) if isinstance(v, str)]
    growth = [str(v) for v in soul.get("growth_notes", []) if isinstance(v, str)]
    style = soul.get("style_preferences", {})
    return _render(
        _read_template("evolving_soul"),
        {
            "traits": _list_to_bullets(traits),
            "user_alignment_notes": _list_to_bullets(align[-5:]),
            "growth_notes": _list_to_bullets(growth[-5:]),
            "style_preferences": json.dumps(style),
        },
    )


def build_system_prompt(memory: MemoryStore, soul_rules: str, session_id: str) -> str:
    """Build full system prompt from templates and persisted memory."""
    _ = session_id
    profile = memory.load_profile()
    preferences = memory.load_preferences()
    projects = memory.load_projects()
    todos = memory.load_todos()

    sections: list[str] = []
    sections.append(_render(_read_template("soul"), {"soul_rules": soul_rules}))
    sections.append(_read_template("core_persona").strip())
    sections.append(_read_template("intelligence_contract").strip())
    sections.append(_read_template("anti_chatbot_contract").strip())
    sections.append(_evolving_soul_card(memory))
    sections.append(
        _render(
            _read_template("identity"),
            {
                "name": str(profile.get("name", "") or "(unknown)"),
                "environment_summary": str(profile.get("environment_summary", "") or "(unknown)"),
                "long_term_goals": _list_to_bullets([str(v) for v in profile.get("long_term_goals", []) if isinstance(v, str)]),
                "behavioral_constraints": _list_to_bullets(
                    [str(v) for v in profile.get("behavioral_constraints", []) if isinstance(v, str)]
                ),
            },
        )
    )
    sections.append(
        _render(
            _read_template("prefs"),
            {
                "response_style": str(preferences.get("response_style", "concise")),
                "autonomy_level": str(preferences.get("autonomy_level", "observe")),
                "quiet_hours": json.dumps(preferences.get("quiet_hours", ["23:00", "08:00"])),
                "notification_limits": json.dumps(preferences.get("notification_limits", {"max_per_day": 5})),
            },
        )
    )
    sections.append(_render(_read_template("projects"), {"project_cards": _project_cards(projects)}))
    sections.append(_render(_read_template("todos"), {"todo_cards": _todo_cards(todos)}))
    sections.append(_render(_read_template("facts"), {"fact_snippets": _fact_snippets(memory)}))
    sections.append(_read_template("living_persona").strip())

    return "\n\n".join(section for section in sections if section)
