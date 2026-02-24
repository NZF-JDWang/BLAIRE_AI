from __future__ import annotations

from pathlib import Path

from blaire_core.learning.soul_growth import apply_soul_growth_updates
from blaire_core.memory.store import MemoryStore
from blaire_core.prompting.composer import build_system_prompt


def test_evolving_soul_defaults_and_prompt_section(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    soul = store.load_evolving_soul()
    prompt = build_system_prompt(store, soul_rules="Be useful.", session_id="s1")

    assert soul["version"] == 1
    assert isinstance(soul.get("traits"), list)
    assert "# Evolving Soul (Living Layer)" in prompt


def test_soul_growth_updates_and_reset(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    updates = apply_soul_growth_updates(
        store,
        user_message="That helped a lot, keep doing this concise structure.",
        assistant_message="I will keep this structure.",
    )
    soul = store.load_evolving_soul()

    assert updates["updated"] is True
    assert updates["alignment_notes_added"] >= 1
    assert updates["growth_notes_added"] >= 1
    assert soul["user_alignment_notes"]
    assert soul["growth_notes"]

    reset = store.reset_evolving_soul()
    assert reset["user_alignment_notes"] == []
    assert reset["growth_notes"] == []
