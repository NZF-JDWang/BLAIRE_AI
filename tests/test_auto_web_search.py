from __future__ import annotations

from dataclasses import replace

from blaire_core.config import read_config_snapshot
from blaire_core.orchestrator import _should_auto_web_search, build_context, handle_user_message


def test_should_auto_web_search_patterns() -> None:
    assert _should_auto_web_search("latest ollama release notes")
    assert _should_auto_web_search("What is the current weather in Wellington?")
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
