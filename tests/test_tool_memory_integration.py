from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from blaire_core.learning.tool_memory import distill_tool_result_to_memory
from blaire_core.memory.store import MemoryStore
from blaire_core.orchestrator import call_tool
from blaire_core.tools.registry import Tool, ToolRegistry


def _all_events(store: MemoryStore) -> list[dict]:
    return store.structured.get_events_since("1970-01-01T00:00:00+00:00", limit=200)


def _context_with_tool(store: MemoryStore, name: str, fn) -> SimpleNamespace:
    registry = ToolRegistry()
    registry.register(Tool(name=name, description="test tool", risk_level="safe", fn=fn))
    return SimpleNamespace(memory=store, tools=registry)


def test_distill_tool_result_deduplicates_similar_facts(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    result = {
        "ok": True,
        "tool": "web_search",
        "data": {
            "query": "best python formatter",
            "results": [
                {"title": "Ruff formatter", "url": "https://example.org/ruff", "snippet": "formatter"},
            ],
        },
        "metadata": {},
    }

    first = distill_tool_result_to_memory(memory=store, tool_name="web_search", args={"query": "best python formatter"}, result=result)
    second = distill_tool_result_to_memory(memory=store, tool_name="web_search", args={"query": "best python formatter"}, result=result)

    memories = store.get_memories(tags=["tool:web_search"], limit=20)

    assert first["written"] == 1
    assert second["written"] == 0
    assert len(memories) == 1


def test_call_tool_failed_does_not_persist_memory(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    def _failing(_args: dict) -> dict:
        return {
            "ok": False,
            "tool": "web_search",
            "data": None,
            "error": {"code": "request_failed", "message": "timeout"},
            "metadata": {},
        }

    context = _context_with_tool(store, "web_search", _failing)
    result = call_tool(context, name="web_search", args={"query": "latest rust release"})

    assert result["ok"] is False
    assert store.get_memories(tags=["tool:web_search"], limit=20) == []

    events = _all_events(store)
    assert [event["event_type"] for event in events] == ["tool_call_started", "tool_call_failed"]


def test_call_tool_success_persists_distilled_fact(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path))
    store.initialize()

    def _ok_tool(args: dict) -> dict:
        return {
            "ok": True,
            "tool": "web_search",
            "data": {
                "query": str(args.get("query", "")),
                "results": [
                    {
                        "title": "OpenAI announces updates",
                        "url": "https://example.org/openai-updates",
                        "snippet": "external snippet",
                    }
                ],
            },
            "metadata": {"cached": False},
        }

    context = _context_with_tool(store, "web_search", _ok_tool)
    result = call_tool(context, name="web_search", args={"query": "OpenAI updates"})

    assert result["ok"] is True
    memories = store.get_memories(tags=["tool:web_search", "source:external"], limit=20)
    assert len(memories) == 1
    assert "Web lookup for 'OpenAI updates'" in memories[0]["text"]

    events = _all_events(store)
    assert [event["event_type"] for event in events] == ["tool_call_started", "tool_call_finished"]
    assert events[-1]["payload"]["summary"]["memory_distilled"]["written"] == 1
