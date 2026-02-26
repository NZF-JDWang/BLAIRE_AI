"""CLI REPL interface."""

from __future__ import annotations

import json
import shlex
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from typing import Callable
import re

from blaire_core.memory.store import clean_stale_locks
from blaire_core.notifications import notify_user, notify_user_media
from blaire_core.orchestrator import (
    AppContext,
    call_tool,
    diagnostics,
    handle_user_message,
    handle_user_message_with_tool,
    health_summary_quick,
)
from blaire_core.telegram_bridge import TelegramTextBridge, process_telegram_updates
from blaire_core.telegram_client import get_telegram_updates


RESTRICTED_ALLOWED_PREFIXES = {
    "/help",
    "/exit",
    "/health",
    "/admin status",
    "/admin config",
    "/admin diagnostics",
    "/admin selfcheck",
}


def _is_allowed_restricted(line: str) -> bool:
    normalized = line.strip().lower()
    return any(normalized.startswith(prefix) for prefix in RESTRICTED_ALLOWED_PREFIXES)


@dataclass(slots=True)
class CliState:
    active_session_id: str


@dataclass(slots=True)
class ExplicitToolIntent:
    tool_name: str
    args: dict[str, object]


def parse_explicit_tool_intent(user_message: str) -> ExplicitToolIntent | None:
    """Parse lightweight user intents that should route to direct tool invocation."""
    lowered = user_message.strip().lower()
    if not lowered:
        return None

    disk_patterns = (
        r"\b(check|show|tell me)\b.{0,20}\b(disk|storage)\b.{0,20}\b(space|usage|status)\b",
        r"\bhow much\b.{0,20}\b(disk|storage)\b.{0,20}\b(left|free|available)\b",
    )
    if any(re.search(pattern, lowered) for pattern in disk_patterns):
        path_match = re.search(r"\b(?:on|for|at)\s+(/[^\s]+)", user_message)
        return ExplicitToolIntent(
            tool_name="check_disk_space",
            args={"path": path_match.group(1) if path_match else "."},
        )

    if re.search(r"\b(search|find|look up|lookup)\b.{0,20}\b(notes|memory|memories)\b", lowered):
        query = re.sub(r"^\s*(?:can you\s+|please\s+)?(?:search|find|look up|lookup)\s+(?:my\s+)?(?:notes|memory|memories)\s*(?:for\s+)?", "", lowered).strip(" ?.!")
        return ExplicitToolIntent(
            tool_name="local_search",
            args={"query": query or "notes", "limit": 5},
        )

    if re.search(r"\bups\b", lowered) and re.search(r"\b(status|tracking|update|where)\b", lowered):
        cleaned = re.sub(r"\b(?:what(?:'s| is)|check|tell me|show me)\b", "", user_message, flags=re.IGNORECASE)
        query = cleaned.strip(" ?.!") or "UPS status"
        return ExplicitToolIntent(tool_name="web_search", args={"query": query, "count": 3})

    return None


def _print_help() -> None:
    print("Commands:")
    print("/help")
    print("/exit")
    print("/health")
    print("/heartbeat tick|start|stop|status")
    print('/telegram test "<message>"')
    print("/telegram listen")
    print("/telegram start|stop|status")
    print("/telegram send-voice <path> [caption]")
    print("/telegram send-audio <path> [caption]")
    print("/telegram send-file <path> [caption]")
    print("/tool <name> <json_args>  (admin/debug: direct raw tool call)")
    print("/brain soul|rules|user|memory|heartbeat|style|edit <file>")
    print("/session new|list|use|current")
    print("/session cleanup --dry-run|--enforce [--active-key <id>]")
    print("/admin status|config [--effective]|diagnostics [--deep]|selfcheck|memory [stats|recent|patterns|search]|soul [state|--reset]")
    print("Tip: plain language requests (e.g. 'check disk space', 'search my notes', 'what\'s UPS status') auto-route tools.")


def _parse_limit(tokens: list[str], default: int) -> int:
    lowered = [token.lower() for token in tokens]
    if "--limit" not in lowered:
        return default
    try:
        idx = lowered.index("--limit")
        if idx + 1 < len(tokens):
            return max(1, int(tokens[idx + 1]))
    except (ValueError, TypeError):
        return default
    return default


def _build_soul_state(context: AppContext) -> dict[str, object]:
    soul = context.memory.load_evolving_soul()
    traits = [str(v) for v in soul.get("traits", []) if isinstance(v, str)]
    alignment_notes = [str(v) for v in soul.get("user_alignment_notes", []) if isinstance(v, str)]
    growth_notes = [str(v) for v in soul.get("growth_notes", []) if isinstance(v, str)]
    capability_memories = context.memory.get_memories(tags=["system", "capability"], limit=5)
    preference_memories = context.memory.get_memories(memory_type="preference", limit=5)
    top_patterns = context.memory.get_top_patterns(limit=3)
    return {
        "traits": traits[:8],
        "style_preferences": soul.get("style_preferences", {}),
        "recent_alignment_notes": alignment_notes[-5:],
        "recent_growth_notes": growth_notes[-5:],
        "recent_capability_memories": [
            {"text": str(item.get("text", "")), "tags": item.get("tags", []), "importance": item.get("importance", 0)}
            for item in capability_memories[:5]
        ],
        "recent_preference_memories": [
            {"text": str(item.get("text", "")), "tags": item.get("tags", []), "importance": item.get("importance", 0)}
            for item in preference_memories[:5]
        ],
        "top_patterns": [
            {"text": str(item.get("text", "")), "tags": item.get("tags", []), "importance": item.get("importance", 0)}
            for item in top_patterns[:3]
        ],
    }


def _handle_heartbeat(context: AppContext, tokens: list[str]) -> None:
    if len(tokens) < 2:
        print("Usage: /heartbeat tick|start|stop|status")
        return
    command = tokens[1].lower()
    if command == "tick":
        context.heartbeat.tick_once()
        print("Heartbeat tick executed.")
    elif command == "start":
        context.heartbeat.start()
        print("Heartbeat loop started.")
    elif command == "stop":
        context.heartbeat.stop()
        print("Heartbeat loop stopped.")
    elif command == "status":
        status = context.heartbeat.status()
        print(json.dumps(status.__dict__, indent=2))
    else:
        print("Unknown heartbeat subcommand.")


def _handle_admin(context: AppContext, tokens: list[str]) -> None:
    if len(tokens) < 2:
        print("Usage: /admin status|config|diagnostics|memory|soul [--reset]")
        return
    sub = tokens[1].lower()
    if sub == "status":
        print(health_summary_quick(context))
    elif sub == "config":
        redacted = context.config_snapshot.path
        issues = context.config_snapshot.issues
        payload: dict[str, object] = {"path": redacted, "valid": context.config_snapshot.valid, "issues": issues}
        if any(token.lower() == "--effective" for token in tokens[2:]):
            if context.config_snapshot.effective_raw is not None:
                payload["effective_config"] = context.config_snapshot.effective_raw
                payload["effective_source"] = "snapshot_merged"
            else:
                payload["effective_config"] = asdict(context.config)
                payload["effective_source"] = "runtime_fallback"
        print(json.dumps(payload, indent=2))
    elif sub == "diagnostics":
        deep = any(token.lower() == "--deep" for token in tokens[2:])
        print(json.dumps(diagnostics(context, deep=deep), indent=2))
    elif sub == "selfcheck":
        llm_ok, llm_detail = context.llm.check_reachable(timeout_seconds=3)
        heartbeat_status = context.heartbeat.status()
        telegram = context.config.telegram
        web_tool = context.tools.get("web_search")
        web_ready = bool(context.config.tools.web_search.api_key or __import__("os").getenv("BLAIRE_BRAVE_API_KEY"))
        web_probe: dict[str, object] = {"checked": False}
        if web_tool and web_ready:
            try:
                probe = web_tool.fn({"query": "BLAIRE selfcheck", "count": 1})
                web_probe = {
                    "checked": True,
                    "ok": bool(probe.get("ok")),
                    "error": probe.get("error"),
                }
            except Exception as exc:  # noqa: BLE001
                web_probe = {"checked": True, "ok": False, "error": str(exc)}
        payload = {
            "config_valid": context.config_snapshot.valid,
            "llm": {"ok": llm_ok, "detail": llm_detail},
            "heartbeat": {"running": heartbeat_status.running, "interval_seconds": heartbeat_status.interval_seconds},
            "memory": context.memory.structured.get_stats(),
            "telegram": {
                "enabled": telegram.enabled,
                "polling_enabled": telegram.polling_enabled,
                "has_bot_token": bool(telegram.bot_token),
                "has_chat_id": bool(telegram.chat_id),
            },
            "web_search": {"ready": web_ready, "probe": web_probe},
        }
        print(json.dumps(payload, indent=2))
    elif sub == "memory":
        if len(tokens) == 2 or tokens[2].lower() == "stats":
            sessions = list(context.memory.sessions_dir.glob("session-*.json"))
            long_term_files = list(context.memory.long_term_dir.glob("*.jsonl"))
            summary = {
                "sessions_count": len(sessions),
                "session_total_bytes": sum(p.stat().st_size for p in sessions),
                "long_term_files": [str(p) for p in long_term_files],
                "long_term_total_bytes": sum(p.stat().st_size for p in long_term_files),
                "structured": context.memory.structured.get_stats(),
            }
            print(json.dumps(summary, indent=2))
            return
        memory_sub = tokens[2].lower()
        if memory_sub == "recent":
            limit = _parse_limit(tokens[3:], default=10)
            print(json.dumps(context.memory.get_memories(limit=limit), indent=2))
            return
        if memory_sub == "patterns":
            limit = _parse_limit(tokens[3:], default=5)
            print(json.dumps(context.memory.get_top_patterns(limit=limit), indent=2))
            return
        if memory_sub == "search":
            query = " ".join(tokens[3:]).strip()
            if not query:
                print("Usage: /admin memory search <query>")
                return
            print(json.dumps(context.memory.retrieve_relevant_memories(query=query, limit=10), indent=2))
            return
        print("Usage: /admin memory stats|recent [--limit N]|patterns [--limit N]|search <query>")
    elif sub == "soul":
        if len(tokens) >= 3 and tokens[2].lower() == "state":
            print(json.dumps(_build_soul_state(context), indent=2))
            return
        if any(token.lower() == "--reset" for token in tokens[2:]):
            soul = context.memory.reset_evolving_soul()
            print(json.dumps({"reset": True, "soul": soul}, indent=2))
            return
        print(json.dumps(context.memory.load_evolving_soul(), indent=2))
    else:
        print("Unknown admin subcommand.")



_BRAIN_FILE_MAP = {
    "soul": "SOUL.md",
    "rules": "RULES.md",
    "user": "USER.md",
    "memory": "MEMORY.md",
    "heartbeat": "HEARTBEAT.md",
    "style": "STYLE.md",
}


def _handle_brain(context: AppContext, tokens: list[str]) -> None:
    if len(tokens) < 2:
        print("Usage: /brain soul|rules|user|memory|heartbeat|style|edit <file>")
        return

    command = tokens[1].lower()
    if command == "edit":
        if len(tokens) < 3:
            print("Usage: /brain edit <file>")
            return
        command = tokens[2].lower()

    filename = _BRAIN_FILE_MAP.get(command)
    if not filename:
        print("Unknown brain file. Use: soul, rules, user, memory, heartbeat, style")
        return

    context.brain_composer.ensure_brain_files()
    path = context.brain_composer.brain_dir / filename
    if not path.exists():
        print(f"Brain file missing: {path}")
        return

    print(f"# {filename} ({path})")
    print(path.read_text(encoding="utf-8"))

def _handle_tool(context: AppContext, tokens: list[str]) -> None:
    if len(tokens) < 2:
        print("Usage: /tool <name> <json_args>")
        return
    name = tokens[1]
    args = {}
    if len(tokens) > 2:
        try:
            args = json.loads(" ".join(tokens[2:]))
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON args: {exc}")
            return
    result = call_tool(context, name=name, args=args)
    print(json.dumps(result, indent=2))


def _handle_session(context: AppContext, state: CliState, tokens: list[str]) -> None:
    if len(tokens) < 2:
        print("Usage: /session new|list|use|current|cleanup")
        return
    sub = tokens[1].lower()
    if sub == "new":
        state.active_session_id = str(uuid.uuid4())
        context.memory.load_or_create_session(state.active_session_id)
        print(f"New session: {state.active_session_id}")
    elif sub == "list":
        sessions = sorted(context.memory.sessions_dir.glob("session-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in sessions:
            print(path.stem.replace("session-", ""))
    elif sub == "use":
        if len(tokens) < 3:
            print("Usage: /session use <id>")
            return
        state.active_session_id = tokens[2]
        context.memory.load_or_create_session(state.active_session_id)
        print(f"Active session: {state.active_session_id}")
    elif sub == "current":
        print(state.active_session_id)
    elif sub == "cleanup":
        dry_run = "--dry-run" in [t.lower() for t in tokens[2:]]
        enforce = "--enforce" in [t.lower() for t in tokens[2:]]
        active_key: str | None = None
        if "--active-key" in [t.lower() for t in tokens[2:]]:
            for idx, token in enumerate(tokens):
                if token.lower() == "--active-key" and idx + 1 < len(tokens):
                    active_key = tokens[idx + 1]
        if not dry_run and not enforce:
            print("Use /session cleanup --dry-run or /session cleanup --enforce")
            return
        preview = context.memory.preview_session_cleanup(
            mode=context.config.session.maintenance.mode,
            prune_after=context.config.session.maintenance.prune_after,
            max_entries=context.config.session.maintenance.max_entries,
            max_disk_bytes=context.config.session.maintenance.max_disk_bytes,
            high_water_ratio=context.config.session.maintenance.high_water_ratio,
            active_key=active_key,
        )
        if dry_run:
            print(json.dumps({"dry_run": True, "preview": preview}, indent=2))
            return
        result = context.memory.enforce_session_cleanup(preview)
        print(json.dumps({"dry_run": False, "preview": preview, "result": result}, indent=2))
    else:
        print("Unknown session subcommand.")


def _handle_health(context: AppContext) -> None:
    print(health_summary_quick(context))


def _handle_telegram(context: AppContext, tokens: list[str], bridge: TelegramTextBridge | None = None) -> int:
    if len(tokens) < 2:
        print('Usage: /telegram test "<message>"|listen|start|stop|status|send-voice|send-audio|send-file')
        return 2
    sub = tokens[1].lower()
    if sub == "listen":
        return _handle_telegram_listen(context)
    if sub == "start":
        if bridge is None:
            print("Telegram bridge control is only available in interactive mode.")
            return 2
        if bridge.start():
            print("Telegram polling started.")
            return 0
        print("Telegram polling did not start. Check telegram.enabled, token/chat_id, and polling_enabled.")
        return 2
    if sub == "stop":
        if bridge is None:
            print("Telegram bridge control is only available in interactive mode.")
            return 2
        bridge.stop()
        print("Telegram polling stopped.")
        return 0
    if sub == "status":
        if bridge is None:
            print("Telegram bridge control is only available in interactive mode.")
            return 2
        print(json.dumps(bridge.status().__dict__, indent=2))
        return 0
    if sub in {"send-voice", "send-audio", "send-file"}:
        if len(tokens) < 3:
            print(f"Usage: /telegram {sub} <path> [caption]")
            return 2
        path = tokens[2]
        caption = " ".join(tokens[3:]).strip() or None
        media_map = {"send-voice": "voice", "send-audio": "audio", "send-file": "document"}
        sent = notify_user_media(context.config, path, media_type=media_map[sub], caption=caption, via_telegram=True)
        if not context.config.telegram.enabled:
            print("Telegram is disabled in configuration; enable BLAIRE_TELEGRAM_ENABLED to send media.")
            return 0
        if sent:
            print("Telegram media sent.")
            return 0
        print("Telegram media send failed.")
        return 1
    if sub != "test" or len(tokens) < 3:
        print('Usage: /telegram test "<message>"|listen|start|stop|status|send-voice|send-audio|send-file')
        return 2

    message = " ".join(tokens[2:]).strip()
    if not message:
        print("Message cannot be empty.")
        return 2

    sent = notify_user(context.config, message, level="info", via_telegram=True)
    if not context.config.telegram.enabled:
        print("Telegram is disabled in configuration; enable BLAIRE_TELEGRAM_ENABLED to send messages.")
        return 0
    if sent:
        print("Telegram test message sent.")
        return 0
    print("Telegram send failed. Check bot token/chat id and network connectivity.")
    return 1


def _handle_telegram_listen(context: AppContext) -> int:
    telegram = context.config.telegram
    if not telegram.enabled or not telegram.bot_token or not telegram.chat_id:
        print("Telegram is disabled or misconfigured; set enabled, bot token, and chat id.")
        return 2

    offset: int | None = None
    print("Telegram listener started. Ctrl+C to stop.")
    try:
        while True:
            updates, offset = get_telegram_updates(
                telegram.bot_token,
                offset=offset,
                timeout=telegram.polling_timeout_seconds,
            )
            process_telegram_updates(context, updates)
    except KeyboardInterrupt:
        print("\nTelegram listener stopped.")
        return 0


def execute_single_command(context: AppContext, command_line: str, initial_session_id: str | None = None) -> int:
    state = CliState(active_session_id=initial_session_id or str(uuid.uuid4()))
    context.memory.load_or_create_session(state.active_session_id)
    try:
        tokens = shlex.split(command_line)
    except ValueError as exc:
        print(f"Parse error: {exc}")
        return 2
    if not tokens:
        print("Empty command.")
        return 2
    command = tokens[0].lower()
    if command == "/telegram":
        return _handle_telegram(context, tokens, None)
    if command == "/admin":
        _handle_admin(context, tokens)
        return 0
    if command == "/health":
        _handle_health(context)
        return 0
    if command == "/heartbeat":
        _handle_heartbeat(context, tokens)
        return 0
    if command == "/session":
        _handle_session(context, state, tokens)
        return 0
    if command == "/tool":
        _handle_tool(context, tokens)
        return 0
    if command == "/brain":
        _handle_brain(context, tokens)
        return 0
    if command == "/help":
        _print_help()
        return 0
    intent = parse_explicit_tool_intent(command_line)
    if intent:
        answer = handle_user_message_with_tool(
            context,
            session_id=state.active_session_id,
            user_message=command_line,
            tool_name=intent.tool_name,
            args=intent.args,
            debug_mode=False,
        )
        print(answer)
        return 0
    answer = handle_user_message(context, session_id=state.active_session_id, user_message=command_line)
    print(answer)
    return 0


def run_cli(context: AppContext, initial_session_id: str | None = None) -> None:
    """Run CLI REPL."""
    state = CliState(active_session_id=initial_session_id or str(uuid.uuid4()))
    telegram_bridge = TelegramTextBridge(context)
    context.memory.load_or_create_session(state.active_session_id)
    if context.config.heartbeat.interval_seconds > 0:
        context.heartbeat.start()
    if context.config.telegram.polling_enabled:
        started = telegram_bridge.start()
        if started:
            print("Telegram polling enabled and started.")
        else:
            print("Telegram polling is enabled in config but could not start.")

    print("BLAIRE CLI ready. Type /help for commands.")
    if not context.config_snapshot.valid:
        print("Config invalid. Entering diagnostics-only mode.")
        for issue in context.config_snapshot.issues:
            print(f"- {issue}")

    handlers: dict[str, Callable[[list[str]], None]] = {
        "/help": lambda _: _print_help(),
        "/exit": lambda _: (_ for _ in ()).throw(EOFError()),
        "/health": lambda _: _handle_health(context),
        "/heartbeat": lambda tokens: _handle_heartbeat(context, tokens),
        "/admin": lambda tokens: _handle_admin(context, tokens),
        "/tool": lambda tokens: _handle_tool(context, tokens),
        "/brain": lambda tokens: _handle_brain(context, tokens),
        "/session": lambda tokens: _handle_session(context, state, tokens),
        "/telegram": lambda tokens: _handle_telegram(context, tokens, telegram_bridge),
    }

    try:
        while True:
            raw = input("> ").strip()
            if not raw:
                continue
            if not context.config_snapshot.valid and raw.startswith("/") and not _is_allowed_restricted(raw):
                print("Config invalid. Command blocked. Allowed: /health, /admin status|config|diagnostics, /help, /exit")
                continue

            if raw.startswith("/"):
                try:
                    tokens = shlex.split(raw)
                except ValueError as exc:
                    print(f"Parse error: {exc}")
                    continue
                if not tokens:
                    continue
                handler = handlers.get(tokens[0].lower())
                if not handler:
                    print("Unknown command. Use /help.")
                    continue
                try:
                    handler(tokens)
                except EOFError:
                    break
                continue

            intent = parse_explicit_tool_intent(raw)
            if intent:
                response = handle_user_message_with_tool(
                    context,
                    session_id=state.active_session_id,
                    user_message=raw,
                    tool_name=intent.tool_name,
                    args=intent.args,
                    debug_mode=False,
                )
            else:
                response = handle_user_message(context, session_id=state.active_session_id, user_message=raw)
            print(response)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        telegram_bridge.stop()
        context.heartbeat.stop()
        scan = clean_stale_locks(context.config.paths.data_root)
        if scan.removed:
            print(f"Cleaned stale locks: {scan.removed}")
