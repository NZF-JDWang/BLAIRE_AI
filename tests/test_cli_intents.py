from __future__ import annotations

from blaire_core.config import read_config_snapshot
from blaire_core.interfaces import cli
from blaire_core.orchestrator import build_context, handle_user_message_with_tool


def test_parse_explicit_tool_intent_disk_defaults() -> None:
    intent = cli.parse_explicit_tool_intent("check disk space")
    assert intent is not None
    assert intent.tool_name == "check_disk_space"
    assert intent.args == {"path": "."}


def test_parse_explicit_tool_intent_local_search_defaults() -> None:
    intent = cli.parse_explicit_tool_intent("search my notes")
    assert intent is not None
    assert intent.tool_name == "local_search"
    assert intent.args == {"query": "notes", "limit": 5}


def test_parse_explicit_tool_intent_web_status_defaults() -> None:
    intent = cli.parse_explicit_tool_intent("what's UPS status")
    assert intent is not None
    assert intent.tool_name == "web_search"
    assert intent.args == {"query": "UPS status", "count": 3}


def test_execute_single_command_routes_explicit_intent(monkeypatch, tmp_path, capsys) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path), "llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    captured: dict[str, object] = {}

    def _fake_routed(context_arg, session_id: str, user_message: str, tool_name: str, args: dict, debug_mode: bool = False) -> str:
        _ = context_arg, session_id
        captured["user_message"] = user_message
        captured["tool_name"] = tool_name
        captured["args"] = args
        captured["debug_mode"] = debug_mode
        return "I checked your disk usage and you're in good shape."

    monkeypatch.setattr(cli, "handle_user_message_with_tool", _fake_routed)

    code = cli.execute_single_command(context, "check disk space")

    assert code == 0
    assert captured["tool_name"] == "check_disk_space"
    assert captured["args"] == {"path": "."}
    assert captured["debug_mode"] is False
    assert "good shape" in capsys.readouterr().out


def test_handle_user_message_with_tool_hides_raw_json_by_default(monkeypatch, tmp_path) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path), "llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    def _fake_local_search(args: dict) -> dict:
        return {
            "ok": True,
            "tool": "local_search",
            "data": {"query": args["query"], "results": [{"text": "note 1"}]},
            "error": None,
            "metadata": {},
        }

    context.tools.get("local_search").fn = _fake_local_search  # type: ignore[union-attr]
    monkeypatch.setattr(
        context.llm,
        "generate",
        lambda system_prompt, messages, max_tokens: "I searched your notes and found one matching entry: note 1.",
    )

    answer = handle_user_message_with_tool(
        context,
        session_id="s-intent-voice",
        user_message="search my notes for note 1",
        tool_name="local_search",
        args={"query": "note 1", "limit": 5},
        debug_mode=False,
    )

    assert "found one matching entry" in answer.lower()
    assert '"ok"' not in answer


def test_handle_user_message_with_tool_debug_returns_json(tmp_path) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path), "llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    answer = handle_user_message_with_tool(
        context,
        session_id="s-intent-debug",
        user_message="check disk space",
        tool_name="check_disk_space",
        args={"path": "."},
        debug_mode=True,
    )

    assert '"tool": "check_disk_space"' in answer
    assert '"ok": true' in answer.lower()


def test_help_brain_command_does_not_advertise_edit(monkeypatch, tmp_path, capsys) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path), "llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    code = cli.execute_single_command(context, "/help")

    assert code == 0
    output = capsys.readouterr().out
    assert "/brain soul|rules|user|memory|heartbeat|style" in output
    assert "edit <file>" not in output


def test_approvals_commands_roundtrip(monkeypatch, tmp_path, capsys) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path), "llm.model": "test-model"})
    assert snapshot.effective_config is not None
    context = build_context(snapshot.effective_config, snapshot)

    monkeypatch.setattr(
        cli,
        "approve_tool_call",
        lambda context, token, tool_name, args: {
            "ok": True,
            "tool": tool_name,
            "metadata": {"token": token},
            "data": args,
            "error": None,
        },
    )

    list_code = cli.execute_single_command(context, "/approvals list")
    approve_code = cli.execute_single_command(
        context, "/approve tok-1 docker_container_restart '{\"host\":\"bsl1\",\"container\":\"jellyfin\"}'"
    )

    assert list_code == 0
    assert approve_code == 0
    output = capsys.readouterr().out
    assert "pending" in output
    assert "docker_container_restart" in output
