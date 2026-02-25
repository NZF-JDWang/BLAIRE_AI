from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from blaire_core.config import read_config_snapshot
from blaire_core.interfaces.cli import _handle_admin
from blaire_core.orchestrator import build_context, diagnostics, health_summary_quick


def test_health_quick_and_deep() -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)
    quick = health_summary_quick(context)
    assert "config=ok" in quick
    deep = diagnostics(context, deep=True)
    assert deep["deep"] is True
    assert "llm" in deep


def test_admin_selfcheck_output(monkeypatch, capsys) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    monkeypatch.setattr(context.llm, "check_reachable", lambda timeout_seconds=3: (True, "ok"))
    web_tool = context.tools.get("web_search")
    assert web_tool is not None
    monkeypatch.setattr(web_tool, "fn", lambda args: {"ok": True, "data": {"query": args.get("query", "")}})

    _handle_admin(context, ["/admin", "selfcheck"])

    payload = json.loads(capsys.readouterr().out)
    assert "config_valid" in payload
    assert payload["llm"]["ok"] is True
    assert "memory" in payload
    assert "telegram" in payload
    assert "web_search" in payload
