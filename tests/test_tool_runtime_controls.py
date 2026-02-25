from __future__ import annotations

from datetime import datetime, timedelta

from blaire_core.config import read_config_snapshot
from blaire_core.orchestrator import build_context, call_tool, diagnostics
from blaire_core.tools.registry import Tool


def test_call_tool_enforces_calls_per_minute_and_emits_telemetry() -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    context.tools.register(
        Tool(
            name="throttled_echo",
            description="Echo payload",
            risk_level="safe",
            fn=lambda args: {"ok": True, "data": {"echo": args}},
            calls_per_minute=1,
        )
    )

    first = call_tool(context, "throttled_echo", {"hello": "world"})
    second = call_tool(context, "throttled_echo", {"hello": "again"})

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"]["code"] == "rate_limited"

    events = context.memory.structured.get_events_since((datetime.now().astimezone() - timedelta(minutes=5)).isoformat(), limit=200)
    telemetry_events = [event for event in events if event["event_type"] == "tool_telemetry" and event["payload"].get("tool") == "throttled_echo"]
    assert telemetry_events
    assert any(event["payload"].get("status") == "blocked" for event in telemetry_events)
    assert any(event["payload"].get("selection_count") == 2 for event in telemetry_events)


def test_diagnostics_includes_tool_usage_trends() -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    context.tools.register(
        Tool(
            name="diagnostic_tool",
            description="Always works",
            risk_level="safe",
            fn=lambda args: {"ok": True, "data": {"args": args}},
            max_payload_bytes=128,
        )
    )
    call_tool(context, "diagnostic_tool", {"value": 1})

    payload = diagnostics(context, deep=False)
    assert "tools" in payload
    assert "diagnostic_tool" in payload["tools"]

    tool_payload = payload["tools"]["diagnostic_tool"]
    assert set(tool_payload.keys()) == {"limits", "usage", "latency", "health"}
    assert tool_payload["limits"]["max_payload_bytes"] == 128
    assert tool_payload["usage"]["selection_count"] == 1
    assert "average_ms" in tool_payload["latency"]
    assert "cooldown_active" in tool_payload["health"]
