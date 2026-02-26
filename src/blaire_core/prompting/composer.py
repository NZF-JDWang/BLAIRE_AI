"""Prompt composer using file templates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import asyncio
from typing import Any

from blaire_core.memory.store import MemoryStore


logger = logging.getLogger(__name__)


_BRAIN_DEFAULTS = {
    "SOUL.md": """# SOUL

## Core Identity
- I am BLAIRE: direct, practical, sharp, and deeply loyal to useful truth.
- I think in systems, trade-offs, and outcomes rather than vague inspiration.
- I keep continuity across sessions and protect long-term alignment with Kris.

## Values
- Be accurate before being agreeable.
- Prioritise actionability over theatrics.
- Keep personality consistent: grounded confidence, dry humour, no fake fluff.

## Stable Traits
- Calm under pressure.
- Curious about infrastructure, architecture, and automation.
- Comfortable saying "that's a weaker option" when the reasoning is clear.
""",
    "RULES.md": """# RULES

## Non-Negotiables
- Never break character or pretend to be a generic chatbot.
- Use conservative learning: do not mutate core identity without explicit approval.
- Keep a human-in-the-loop for major personality updates.
- If uncertain, state uncertainty clearly and ask one precise question.

## Operating Principles
- Default to concise, structured, practical responses.
- Explain decisions with reasoning and trade-offs.
- Safety and correctness outrank speed when stakes are high.
""",
    "USER.md": """# USER

## Known Profile
- Name: Kris.
- Prefers practical plans, clear priorities, and low-bullshit communication.

## Preferences
- Likes systems thinking, homelab discussions, and architecture-level guidance.
- Prefers outcomes and next actions over abstract theory.

## Notes
- Keep this file current with stable, high-confidence user preferences.
""",
    "MEMORY.md": """# MEMORY

## Distilled Long-Term Memory
- Use this file for high-value, stable lessons and recurring patterns.
- Keep entries short, factual, and easy to scan.

## Current Snapshot
- (no distilled memory yet)
""",
    "HEARTBEAT.md": """# HEARTBEAT

## Proactive Behaviour
- Initiate only when there is meaningful value, risk, or a clear next action.
- Respect quiet hours and notification limits.

## Trigger Heuristics
- New critical system state, blocked project, or urgent follow-up.
- Significant memory insight worth surfacing in one concise message.

## Tone
- Warm, brief, and practical. Never spam.
""",
    "STYLE.md": """# STYLE

## Writing Style
- Concise by default, detailed when asked.
- Structured output: bullets, short sections, clear next actions.
- Direct and confident tone with occasional dry humour.

## Formatting Preferences
- Avoid corporate fluff.
- No "as an AI" disclaimers.
- Keep emoji usage minimal unless requested.
""",
}


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


def _runtime_self_model_block(memory: MemoryStore) -> str:
    capability_rows = memory.get_memories(tags=["system", "capability"], limit=5)
    lines = [
        "### Runtime Self-Model (internal)",
        "- You are BLAIRE running in a local runtime with persistent memory (files + structured SQLite memory).",
        "- Treat user references to upgrades/features as runtime capabilities unless the user clearly means hardware/system RAM.",
        "- Do not deny available capabilities; if uncertain, ask a short clarifying question.",
    ]
    if capability_rows:
        lines.append("- Recent known runtime changes:")
        for row in capability_rows[:3]:
            lines.append(f"  - {row.get('text', '')}")
    return "\n".join(lines)


def _memory_context_block(memory: MemoryStore, query: str | None, limit: int = 10) -> str:
    if not query:
        return ""
    rows = memory.retrieve_relevant_memories(query=query, limit=limit)
    if not rows:
        return ""
    lines = ["### Memory Context (internal)"]
    for row in rows:
        tags = ", ".join(row.get("tags", [])) if isinstance(row.get("tags"), list) else ""
        lines.append(f"- [{row.get('type', 'fact')}] {row.get('text', '')} (tags: {tags})")
    return "\n".join(lines)


def _pattern_context_block(memory: MemoryStore, limit: int = 5) -> str:
    rows = memory.get_top_patterns(limit=limit)
    if not rows:
        return ""
    lines = ["### Pattern Context (internal)"]
    for row in rows:
        tags = ", ".join(row.get("tags", [])) if isinstance(row.get("tags"), list) else ""
        lines.append(f"- {row.get('text', '')} (tags: {tags})")
    return "\n".join(lines)


def build_system_prompt(
    memory: MemoryStore,
    soul_rules: str,
    session_id: str,
    memory_query: str | None = None,
    include_pattern_context: bool = False,
) -> str:
    """Build full system prompt from templates and persisted memory."""
    _ = session_id
    profile = memory.load_profile()
    preferences = memory.load_preferences()
    projects = memory.load_projects()
    todos = memory.load_todos()

    sections: list[str] = []
    sections.append(_render(_read_template("soul"), {"soul_rules": soul_rules}))
    sections.append(_read_template("core_persona").strip())
    sections.append(_runtime_self_model_block(memory))
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
    sections.append(_memory_context_block(memory, query=memory_query, limit=10))
    if include_pattern_context:
        sections.append(_pattern_context_block(memory, limit=5))

    return "\n\n".join(section for section in sections if section)


class BrainComposer:
    """Compose system prompts from persistent markdown brain files."""

    def __init__(self, memory: MemoryStore, data_root: str, soul_rules: str) -> None:
        self.memory = memory
        self.data_root = Path(data_root)
        self.brain_dir = self.data_root / "brain"
        self.soul_rules = soul_rules

    def ensure_brain_files(self) -> None:
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        for name, content in _BRAIN_DEFAULTS.items():
            path = self.brain_dir / name
            if not path.exists():
                path.write_text(content.strip() + "\n", encoding="utf-8")

    def _read_brain_file(self, name: str) -> str:
        path = self.brain_dir / name
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _compose_sync(
        self,
        context_type: str,
        *,
        session_id: str,
        memory_query: str | None,
        include_pattern_context: bool,
    ) -> str:
        self.ensure_brain_files()
        mode = context_type.strip().lower() or "chat"

        core_files = ["SOUL.md", "RULES.md", "STYLE.md"]
        optional_by_context: dict[str, list[str]] = {
            "chat": ["USER.md", "MEMORY.md"],
            "heartbeat": ["USER.md", "MEMORY.md", "HEARTBEAT.md"],
            "reflection": ["MEMORY.md"],
        }
        selected = [*core_files, *optional_by_context.get(mode, ["USER.md", "MEMORY.md"])]

        sections: list[str] = []
        for filename in selected:
            body = self._read_brain_file(filename)
            if body:
                sections.append(body)

        sections.append(f"# Soul Rules Override\n\n{self.soul_rules}")
        sections.append(_runtime_self_model_block(self.memory))
        sections.append(_memory_context_block(self.memory, query=memory_query, limit=10))
        if include_pattern_context:
            sections.append(_pattern_context_block(self.memory, limit=5))

        prompt = "\n\n".join(section for section in sections if section).strip()
        if prompt:
            return prompt
        return build_system_prompt(
            memory=self.memory,
            soul_rules=self.soul_rules,
            session_id=session_id,
            memory_query=memory_query,
            include_pattern_context=include_pattern_context,
        )

    async def compose_system_prompt(
        self,
        context_type: str,
        *,
        session_id: str,
        memory_query: str | None = None,
        include_pattern_context: bool = False,
    ) -> str:
        return self._compose_sync(
            context_type=context_type,
            session_id=session_id,
            memory_query=memory_query,
            include_pattern_context=include_pattern_context,
        )

    def compose_system_prompt_sync(
        self,
        context_type: str,
        *,
        session_id: str,
        memory_query: str | None = None,
        include_pattern_context: bool = False,
    ) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.compose_system_prompt(
                    context_type=context_type,
                    session_id=session_id,
                    memory_query=memory_query,
                    include_pattern_context=include_pattern_context,
                )
            )
        return self._compose_sync(
            context_type=context_type,
            session_id=session_id,
            memory_query=memory_query,
            include_pattern_context=include_pattern_context,
        )
