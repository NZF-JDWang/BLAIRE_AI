from __future__ import annotations

from dataclasses import replace

from blaire_core.config import read_config_snapshot
from blaire_core.orchestrator import _should_auto_web_search, build_context, handle_user_message, plan_tool_calls
import blaire_core.orchestrator as orchestrator


def test_should_auto_web_search_patterns() -> None:
    assert _should_auto_web_search("latest ollama release notes")
    assert _should_auto_web_search("What is the current weather in Wellington?")
    assert _should_auto_web_search("can you search the web for qwen release notes?")
    assert not _should_auto_web_search("rewrite this paragraph")


def test_auto_web_search_injected_into_messages(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    cfg = snapshot.effective_config
    cfg = replace(
        cfg,
        tools=replace(
            cfg.tools,
            web_search=replace(cfg.tools.web_search, auto_use=True, auto_count=2),
        ),
    )
    context = build_context(cfg, snapshot)

    captured = {"messages": None, "called": 0}

    def _fake_web(args: dict) -> dict:
        captured["called"] += 1
        return {
            "ok": True,
            "data": {
                "query": args["query"],
                "provider": "brave",
                "results": [{"title": "Example", "url": "https://example.com", "snippet": "External snippet"}],
            },
        }

    def _fake_generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = (system_prompt, max_tokens)
        captured["messages"] = messages
        return "ok"

    context.tools.get("web_search").fn = _fake_web  # type: ignore[union-attr]
    monkeypatch.setattr(context.llm, "generate", _fake_generate)

    handle_user_message(context, session_id="s-auto-web", user_message="latest security news today")

    assert captured["called"] == 1
    assert captured["messages"]
    assert captured["messages"][0]["role"] == "system"
    assert "Web search context" in captured["messages"][0]["content"]


def test_capability_drift_triggers_single_regeneration(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    calls = {"count": 0, "messages": []}

    def _fake_generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = (system_prompt, max_tokens)
        calls["count"] += 1
        calls["messages"].append(messages)
        if calls["count"] == 1:
            return "I don't have internet access, memory beyond this session, and my knowledge is static (2023 cutoff)."
        return "I can use persistent memory and tools here. Tell me the query and I will run web search."

    monkeypatch.setattr(context.llm, "generate", _fake_generate)

    answer = handle_user_message(context, session_id="s-drift", user_message="Can you search the web?")

    assert calls["count"] == 2
    assert "persistent memory" in answer.lower()
    assert any(m.get("role") == "system" and "Capability correction" in m.get("content", "") for m in calls["messages"][1])


def test_capability_drift_falls_back_when_regen_still_denies(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    calls = {"count": 0}

    def _fake_generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = (system_prompt, messages, max_tokens)
        calls["count"] += 1
        return "I don't have internet access, memory beyond this session, and my knowledge is static (2023 cutoff)."

    monkeypatch.setattr(context.llm, "generate", _fake_generate)

    answer = handle_user_message(context, session_id="s-drift-hard", user_message="Can you search the web?")

    assert calls["count"] == 2
    assert "i don't have internet access" not in answer.lower()
    assert "memory beyond this session" not in answer.lower()
    assert "yes." in answer.lower()


def test_memory_recall_prompt_uses_profile_fallback_when_model_claims_memory_disabled(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)
    context.memory.save_profile(
        {
            "name": "JD",
            "environment_summary": "",
            "long_term_goals": ["ship BLAIRE"],
            "behavioral_constraints": [],
        }
    )

    monkeypatch.setattr(
        context.llm,
        "generate",
        lambda system_prompt, messages, max_tokens: "Memory is disabled. I cannot recall your name or prior interactions.",
    )

    answer = handle_user_message(context, session_id="s-recall-fallback", user_message="State my name and goal from memory in one line.")

    assert "your name is jd" in answer.lower()
    assert "ship blaire" in answer.lower()


def test_planner_selects_multiple_candidates() -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    plans = plan_tool_calls(
        context,
        session_id="s-plan",
        user_message="Check current news and my disk storage usage",
        recent_messages=[],
    )

    plan_names = [plan.tool_name for plan in plans]
    assert "web_search" in plan_names
    assert "check_disk_space" in plan_names


def test_planner_prefers_local_search_for_memory_prompt() -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    plans = plan_tool_calls(
        context,
        session_id="s-plan-memory",
        user_message="What did we discuss and store in long-term memory about release plans?",
        recent_messages=[],
    )

    assert any(plan.tool_name == "local_search" for plan in plans)


def test_planner_disabled_uses_regex_web_fallback(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    cfg = replace(
        snapshot.effective_config,
        tools=replace(
            snapshot.effective_config.tools,
            planner=replace(snapshot.effective_config.tools.planner, enabled=False),
        ),
    )
    context = build_context(cfg, snapshot)

    called = {"web": 0}

    def _fake_web(args: dict) -> dict:
        called["web"] += 1
        return {"ok": True, "data": {"query": args["query"], "provider": "brave", "results": []}}

    monkeypatch.setattr(context.llm, "generate", lambda system_prompt, messages, max_tokens: "ok")
    context.tools.get("web_search").fn = _fake_web  # type: ignore[union-attr]

    handle_user_message(context, session_id="s-plan-fallback", user_message="latest release news today")

    assert called["web"] == 1


def test_planned_non_web_tool_uses_call_tool_and_injects_context(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    captured: dict[str, object] = {"messages": None, "tool_name": None}

    def _fake_call_tool(_context, name: str, args: dict) -> dict:
        _ = (_context, args)
        captured["tool_name"] = name
        return {"ok": True, "tool": name, "data": {"summary": "local memory hit"}, "metadata": {}}

    def _fake_generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = (system_prompt, max_tokens)
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr(orchestrator, "call_tool", _fake_call_tool)
    monkeypatch.setattr(context.llm, "generate", _fake_generate)

    handle_user_message(context, session_id="s-plan-local-tool", user_message="What did we discuss from memory?")

    assert captured["tool_name"] == "local_search"
    assert isinstance(captured["messages"], list)
    assert "Tool context (local_search)" in captured["messages"][0]["content"]  # type: ignore[index]


def test_low_confidence_answer_triggers_web_lookup(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    captured = {"web_called": 0, "llm_calls": 0}

    def _fake_web(args: dict) -> dict:
        captured["web_called"] += 1
        return {
            "ok": True,
            "data": {
                "query": args["query"],
                "provider": "brave",
                "results": [{"title": "IIHF", "url": "https://example.com", "snippet": "winner"}],
            },
        }

    def _fake_generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = (system_prompt, messages, max_tokens)
        captured["llm_calls"] += 1
        if captured["llm_calls"] == 1:
            return "I'm not sure who won."
        return "After checking, Team X won."

    context.tools.get("web_search").fn = _fake_web  # type: ignore[union-attr]
    monkeypatch.setattr(context.llm, "generate", _fake_generate)

    answer = handle_user_message(context, session_id="s-low-confidence", user_message="Who won the 2026 winter olympics ice hockey?")

    assert captured["web_called"] == 1
    assert captured["llm_calls"] == 2
    assert "after checking" in answer.lower()


def test_low_confidence_lookup_adds_disclosure_when_missing(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    calls = {"count": 0}

    def _fake_web(args: dict) -> dict:
        _ = args
        return {
            "ok": True,
            "data": {
                "query": "x",
                "provider": "brave",
                "results": [{"title": "Example", "url": "https://example.com", "snippet": "snippet"}],
            },
        }

    def _fake_generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = (system_prompt, messages, max_tokens)
        calls["count"] += 1
        if calls["count"] == 1:
            return "I'm not sure."
        return "Team X won."

    context.tools.get("web_search").fn = _fake_web  # type: ignore[union-attr]
    monkeypatch.setattr(context.llm, "generate", _fake_generate)

    answer = handle_user_message(context, session_id="s-low-confidence-disclose", user_message="Who won?")
    assert calls["count"] == 2
    assert answer.lower().startswith("i wasn't fully sure, so i looked it up.")


def test_time_sensitive_factoid_triggers_web_even_if_first_answer_confident(monkeypatch) -> None:
    snapshot = read_config_snapshot("dev", {"llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    captured = {"web_called": 0, "llm_calls": 0}

    def _fake_web(args: dict) -> dict:
        captured["web_called"] += 1
        return {
            "ok": True,
            "data": {
                "query": args["query"],
                "provider": "brave",
                "results": [{"title": "Official Result", "url": "https://example.com", "snippet": "winner"}],
            },
        }

    def _fake_generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        _ = (system_prompt, messages, max_tokens)
        captured["llm_calls"] += 1
        if captured["llm_calls"] == 1:
            return "No winner exists yet."
        return "The winner was Team X."

    context.tools.get("web_search").fn = _fake_web  # type: ignore[union-attr]
    monkeypatch.setattr(context.llm, "generate", _fake_generate)

    answer = handle_user_message(
        context,
        session_id="s-time-sensitive-factoid",
        user_message="Who won the 2026 Winter Olympics ice hockey?",
    )

    assert captured["web_called"] == 1
    assert captured["llm_calls"] == 2
    assert "looked it up" in answer.lower() or "winner was team x" in answer.lower()
