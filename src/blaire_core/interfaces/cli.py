"""CLI REPL interface."""

from __future__ import annotations

import json
import shlex
import uuid
from dataclasses import dataclass
from typing import Callable

from blaire_core.memory.store import clean_stale_locks
from blaire_core.orchestrator import AppContext, call_tool, diagnostics, handle_user_message, health_summary_quick


RESTRICTED_ALLOWED_PREFIXES = {
    "/help",
    "/exit",
    "/health",
    "/admin status",
    "/admin config",
    "/admin diagnostics",
}


def _is_allowed_restricted(line: str) -> bool:
    normalized = line.strip().lower()
    return any(normalized.startswith(prefix) for prefix in RESTRICTED_ALLOWED_PREFIXES)


@dataclass(slots=True)
class CliState:
    active_session_id: str


def _print_help() -> None:
    print("Commands:")
    print("/help")
    print("/exit")
    print("/health")
    print("/heartbeat tick|start|stop|status")
    print("/tool <name> <json_args>")
    print("/session new|list|use|current")
    print("/session cleanup --dry-run|--enforce [--active-key <id>]")
    print("/admin status|config|diagnostics [--deep]|memory|soul [--reset]")


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
        print(json.dumps({"path": redacted, "valid": context.config_snapshot.valid, "issues": issues}, indent=2))
    elif sub == "diagnostics":
        deep = any(token.lower() == "--deep" for token in tokens[2:])
        print(json.dumps(diagnostics(context, deep=deep), indent=2))
    elif sub == "memory":
        sessions = list(context.memory.sessions_dir.glob("session-*.json"))
        long_term_files = list(context.memory.long_term_dir.glob("*.jsonl"))
        summary = {
            "sessions_count": len(sessions),
            "session_total_bytes": sum(p.stat().st_size for p in sessions),
            "long_term_files": [str(p) for p in long_term_files],
            "long_term_total_bytes": sum(p.stat().st_size for p in long_term_files),
        }
        print(json.dumps(summary, indent=2))
    elif sub == "soul":
        if any(token.lower() == "--reset" for token in tokens[2:]):
            soul = context.memory.reset_evolving_soul()
            print(json.dumps({"reset": True, "soul": soul}, indent=2))
            return
        print(json.dumps(context.memory.load_evolving_soul(), indent=2))
    else:
        print("Unknown admin subcommand.")


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


def run_cli(context: AppContext, initial_session_id: str | None = None) -> None:
    """Run CLI REPL."""
    state = CliState(active_session_id=initial_session_id or str(uuid.uuid4()))
    context.memory.load_or_create_session(state.active_session_id)
    if context.config.heartbeat.interval_seconds > 0:
        context.heartbeat.start()

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
        "/session": lambda tokens: _handle_session(context, state, tokens),
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

            response = handle_user_message(context, session_id=state.active_session_id, user_message=raw)
            print(response)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        context.heartbeat.stop()
        scan = clean_stale_locks(context.config.paths.data_root)
        if scan.removed:
            print(f"Cleaned stale locks: {scan.removed}")
