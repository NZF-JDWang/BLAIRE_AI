from __future__ import annotations

from dataclasses import dataclass

from blaire_core.config import read_config_snapshot
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

