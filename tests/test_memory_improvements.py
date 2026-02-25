from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blaire_core.heartbeat.summarise_events import run_daily_summariser, should_run_daily_summariser
from blaire_core.interfaces.cli import execute_single_command
from blaire_core.memory.store import MemoryStore
from blaire_core.memory_store import StructuredMemoryStore


class _MockLLM:
    def generate(self, system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = system_prompt, messages, max_tokens
        return """```json
        {
          "memories": [
            {"type": "fact", "text": "JD decided to keep Python core.", "importance": 5, "tags": ["architecture"]}
          ],
          "patterns": [
            {"text": "JD often revisits architecture decisions.", "importance": 4, "tags": ["behaviour"]}
          ]
        }
        ```"""


def test_memory_dedup_by_hash(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    store.add_or_update_memory(
        memory_type="fact",
        text="JD decided BLAIRE core stays Python.",
        tags=["architecture"],
        importance=5,
        stability="stable",
    )
    store.add_or_update_memory(
        memory_type="fact",
        text="  jd decided blaire core stays python.  ",
        tags=["arch"],
        importance=3,
        stability="evolving",
    )

    rows = store.get_memories(memory_type="fact", limit=10)
    matches = [row for row in rows if "stays python" in row["text"].lower()]
    assert len(matches) == 1
    assert "architecture" in [tag.lower() for tag in matches[0]["tags"]]


def test_daily_summariser_gating_and_meta(tmp_path: Path) -> None:
    structured = StructuredMemoryStore(tmp_path)
    structured.initialize()

    assert should_run_daily_summariser(str(tmp_path)) is True
    result = run_daily_summariser(str(tmp_path), llm_client=None)
    assert result["processed_events"] == 0
    assert should_run_daily_summariser(str(tmp_path)) is False


def test_summariser_parses_fenced_json_and_writes(tmp_path: Path) -> None:
    structured = StructuredMemoryStore(tmp_path)
    structured.initialize()
    structured.log_event(
        event_type="user_message",
        session_id="s1",
        payload={"content": "We decided to keep Python."},
    )

    result = run_daily_summariser(str(tmp_path), llm_client=_MockLLM())

    assert result["processed_events"] >= 1
    assert result["memories_written"] >= 1
    assert result["patterns_written"] >= 1
    memories = structured.get_memories(limit=20)
    patterns = structured.get_top_patterns(limit=20)
    assert any("python core" in row["text"].lower() for row in memories)
    assert any("architecture decisions" in row["text"].lower() for row in patterns)


def test_event_retention_prunes_old_rows(tmp_path: Path) -> None:
    structured = StructuredMemoryStore(tmp_path)
    structured.initialize()
    structured.log_event(
        event_type="old",
        payload={"content": "old"},
        timestamp="2000-01-01T00:00:00+00:00",
    )
    structured.log_event(
        event_type="new",
        payload={"content": "new"},
        timestamp="2100-01-01T00:00:00+00:00",
    )

    deleted = structured.prune_old_events(keep_days=30)

    assert deleted >= 1
    stats = structured.get_stats()
    assert stats["events"] == 1


def test_execute_single_command_supports_admin_memory(tmp_path: Path, capsys) -> None:
    memory = MemoryStore(str(tmp_path))
    memory.initialize()
    memory.log_event(event_type="user_message", payload={"content": "hello"}, session_id="s1")
    context = SimpleNamespace(memory=memory)

    code = execute_single_command(context, "/admin memory stats")

    assert code == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["structured"]["events"] >= 1
