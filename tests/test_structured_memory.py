from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blaire_core.interfaces.cli import _handle_admin
from blaire_core.learning.routine import apply_learning_updates
from blaire_core.memory.store import MemoryStore
from blaire_core.prompting.composer import build_system_prompt


def test_memory_store_initialization_creates_sqlite_db(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    assert (tmp_path / "memory" / "blaire_memory.db").exists()


def test_learning_promotes_name_and_goal_to_structured_memories(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    apply_learning_updates(
        store,
        user_message="My name is JD. My goal is ship BLAIRE memory system.",
        assistant_message="ack",
    )

    memories = store.get_memories(limit=20)
    assert any(row["type"] == "fact" and "JD's name is JD" in row["text"] for row in memories)
    assert any(row["type"] == "fact" and "long-term goal" in row["text"].lower() for row in memories)


def test_retrieve_relevant_memories_returns_matching_preference(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()
    store.add_or_update_memory(
        memory_type="preference",
        text="JD hates hybrid setups and prefers consistency across stacks.",
        tags=["preferences", "architecture"],
        importance=4,
        stability="evolving",
    )
    store.add_or_update_memory(
        memory_type="fact",
        text="JD uses a Windows dev workstation.",
        tags=["profile"],
        importance=3,
        stability="stable",
    )

    results = store.retrieve_relevant_memories("JD hates hybrid setups", limit=5)

    assert results
    assert "hybrid setups" in results[0]["text"].lower()


def test_prompt_includes_memory_context_block(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()
    store.add_or_update_memory(
        memory_type="decision",
        text="JD decided BLAIRE core will be written in Python.",
        tags=["blaire_core", "architecture"],
        importance=5,
        stability="stable",
    )

    prompt = build_system_prompt(
        store,
        soul_rules="Be useful.",
        session_id="s1",
        memory_query="Should we change BLAIRE language again?",
    )

    assert "### Memory Context (internal)" in prompt
    assert "written in Python" in prompt


def test_admin_memory_subcommands(tmp_path: Path, capsys) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()
    store.log_event("user_message", payload={"content": "hello"}, session_id="s1")
    store.add_or_update_memory(
        memory_type="fact",
        text="JD's name is JD.",
        tags=["profile"],
        importance=5,
        stability="stable",
    )
    store.add_or_update_pattern(
        text="JD tends to refactor architecture when stressed instead of shipping features.",
        source_window="2026-02-24..2026-02-25",
        tags=["behaviour"],
        importance=4,
    )
    context = SimpleNamespace(memory=store)

    _handle_admin(context, ["/admin", "memory", "stats"])
    output_stats = capsys.readouterr().out
    stats_payload = json.loads(output_stats)
    assert stats_payload["structured"]["events"] >= 1
    assert stats_payload["structured"]["memories"] >= 1
    assert stats_payload["structured"]["patterns"] >= 1

    _handle_admin(context, ["/admin", "memory", "recent", "--limit", "1"])
    output_recent = capsys.readouterr().out
    recent_payload = json.loads(output_recent)
    assert len(recent_payload) == 1

    _handle_admin(context, ["/admin", "memory", "patterns", "--limit", "1"])
    output_patterns = capsys.readouterr().out
    patterns_payload = json.loads(output_patterns)
    assert len(patterns_payload) == 1

    _handle_admin(context, ["/admin", "memory", "search", "JD", "name"])
    output_search = capsys.readouterr().out
    search_payload = json.loads(output_search)
    assert search_payload
