from __future__ import annotations

from blaire_core.config import read_config_snapshot
from blaire_core.orchestrator import PlannerAction, build_context, handle_user_message
import blaire_core.orchestrator as orchestrator


def _build_test_context():
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    return build_context(snapshot.effective_config, snapshot)


def test_handle_user_message_single_step_success(monkeypatch) -> None:
    context = _build_test_context()

    calls = {"tool": 0}

    def _plan(*, tool_results, **kwargs):
        if not tool_results:
            return PlannerAction(action="tool", tool_name="local_search", tool_args={"query": "status"}, confidence=0.7)
        return PlannerAction(action="finalize", confidence=1.0)

    def _call_tool(_context, name, args):
        _ = (_context, name, args)
        calls["tool"] += 1
        return {"ok": True, "data": {"text": "done"}}

    monkeypatch.setattr(orchestrator, "_plan_next_action", _plan)
    monkeypatch.setattr(orchestrator, "call_tool", _call_tool)
    monkeypatch.setattr(context.llm, "generate", lambda system_prompt, messages, max_tokens: "single-step answer")

    answer = handle_user_message(context, session_id="loop-single", user_message="do one step")

    assert answer == "single-step answer"
    assert calls["tool"] == 1


def test_handle_user_message_two_step_chain(monkeypatch) -> None:
    context = _build_test_context()

    calls = {"tool": 0}

    def _plan(*, tool_results, **kwargs):
        if len(tool_results) == 0:
            return PlannerAction(action="tool", tool_name="local_search", tool_args={"query": "one"}, confidence=0.6)
        if len(tool_results) == 1:
            return PlannerAction(action="tool", tool_name="check_disk_space", tool_args={}, confidence=0.8)
        return PlannerAction(action="finalize", confidence=1.0)

    def _call_tool(_context, name, args):
        _ = (_context, args)
        calls["tool"] += 1
        return {"ok": True, "tool": name, "data": {"step": calls["tool"]}}

    monkeypatch.setattr(orchestrator, "_plan_next_action", _plan)
    monkeypatch.setattr(orchestrator, "call_tool", _call_tool)
    monkeypatch.setattr(context.llm, "generate", lambda system_prompt, messages, max_tokens: "two-step answer")

    answer = handle_user_message(context, session_id="loop-two", user_message="do two steps")

    assert answer == "two-step answer"
    assert calls["tool"] == 2


def test_handle_user_message_repeated_failure_cutoff(monkeypatch) -> None:
    context = _build_test_context()

    calls = {"tool": 0}

    def _plan(**kwargs):
        return PlannerAction(action="tool", tool_name="web_search", tool_args={"query": "x"}, confidence=0.5)

    def _call_tool(_context, name, args):
        _ = (_context, name, args)
        calls["tool"] += 1
        return {"ok": False, "error": {"code": "timeout", "message": "tool timed out"}}

    monkeypatch.setattr(orchestrator, "_plan_next_action", _plan)
    monkeypatch.setattr(orchestrator, "call_tool", _call_tool)
    monkeypatch.setattr(context.llm, "generate", lambda system_prompt, messages, max_tokens: "fallback after failures")

    answer = handle_user_message(context, session_id="loop-fail", user_message="keep trying")

    assert answer == "fallback after failures"
    assert calls["tool"] == 3


def test_handle_user_message_finalize_without_tools(monkeypatch) -> None:
    context = _build_test_context()

    def _plan(**kwargs):
        return PlannerAction(action="finalize", confidence=0.95, final_answer="planned final")

    monkeypatch.setattr(orchestrator, "_plan_next_action", _plan)
    monkeypatch.setattr(orchestrator, "call_tool", lambda *_args, **_kwargs: {"ok": False})
    monkeypatch.setattr(context.llm, "generate", lambda *_args, **_kwargs: "should not run")

    answer = handle_user_message(context, session_id="loop-final", user_message="no tools")

    assert answer == "planned final"
