from __future__ import annotations

import json
from pathlib import Path

from blaire_core.learning.routine import apply_learning_updates
from blaire_core.memory.store import MemoryStore
from blaire_core.prompting.composer import BrainComposer, build_system_prompt


def test_prompt_composer_includes_template_sections(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()
    store.save_profile(
        {
            "name": "Kris",
            "environment_summary": "Windows dev workstation",
            "long_term_goals": ["Ship BLAIRE v0.1"],
            "behavioral_constraints": ["Be safe"],
        }
    )
    (tmp_path / "projects.json").write_text(
        json.dumps(
            [
                {
                    "id": "p1",
                    "name": "BLAIRE",
                    "description": "Core runtime",
                    "status": "active",
                    "priority": "high",
                    "summary_card": "Implement reliable core",
                    "next_actions": ["tests"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "todos.json").write_text(
        json.dumps(
            [
                {
                    "id": "t1",
                    "project_id": "p1",
                    "title": "Add templates",
                    "description": "",
                    "priority": "now",
                    "status": "open",
                    "created_at": "2026-02-24T00:00:00+00:00",
                    "last_updated": "2026-02-24T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    store.append_fact("user_fact", "User likes practical plans.", ["preference"], 0.8)

    prompt = build_system_prompt(store, soul_rules="Be useful.", session_id="s1")

    assert "# Soul Rules" in prompt
    assert "# Soul Core Persona" in prompt
    assert "### Runtime Self-Model (internal)" in prompt
    assert "# Persona Intelligence Contract" in prompt
    assert "# Anti-Chatbot Contract" in prompt
    assert "# Evolving Soul (Living Layer)" in prompt
    assert "# Identity Card" in prompt
    assert "# Project Cards" in prompt
    assert "# Todo Focus" in prompt
    assert "# Long-Term Memory Snippets" in prompt
    assert "# Soul Living Persona" in prompt
    assert "Ship BLAIRE v0.1" in prompt
    assert "Add templates" in prompt


def test_learning_updates_profile_and_preferences(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    result = apply_learning_updates(
        store,
        user_message="My name is Kris. My goal is ship BLAIRE. Please be detailed.",
        assistant_message="ok",
    )

    profile = store.load_profile()
    prefs = store.load_preferences()
    facts = store.load_facts(limit=10)

    assert profile["name"] == "Kris"
    assert "ship blaire" in [g.lower() for g in profile["long_term_goals"]]
    assert prefs["response_style"] == "detailed"
    assert result["facts_added"] >= 2
    assert any("User name is Kris" in f.get("text", "") for f in facts)


def test_learning_updates_capture_system_upgrade_statements(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    result = apply_learning_updates(
        store,
        user_message="We just upgraded your memory functions and enabled retrieval context.",
        assistant_message="ok",
    )

    memories = store.get_memories(memory_type="decision", limit=20)
    assert result["facts_added"] >= 1
    assert any("system update noted" in m.get("text", "").lower() for m in memories)
    assert any("memory functions" in m.get("text", "").lower() for m in memories)


def test_brain_composer_writes_defaults_and_builds_chat_prompt(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    composer = BrainComposer(memory=store, data_root=str(tmp_path), soul_rules="Stay grounded.")
    prompt = composer.compose_system_prompt_sync(context_type="chat", session_id="s-1")

    assert (tmp_path / "brain" / "SOUL.md").exists()
    assert "# SOUL" in prompt
    assert "# RULES" in prompt
    assert "# STYLE" in prompt
    assert "# USER" in prompt
    assert "# MEMORY" in prompt
