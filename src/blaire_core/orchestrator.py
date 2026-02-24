"""Core orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from blaire_core.config import AppConfig, ConfigSnapshot
from blaire_core.heartbeat.loop import HeartbeatLoop
from blaire_core.llm.client import OllamaClient
from blaire_core.memory.store import MemoryStore, clean_stale_locks
from blaire_core.tools.builtin_tools import (
    check_disk_space,
    check_docker_containers_stub,
    make_local_search_tool,
    make_web_search_tool,
)
from blaire_core.tools.registry import Tool, ToolRegistry


@dataclass(slots=True)
class AppContext:
    config: AppConfig
    config_snapshot: ConfigSnapshot
    memory: MemoryStore
    llm: OllamaClient
    tools: ToolRegistry
    heartbeat: HeartbeatLoop


def build_context(config: AppConfig, snapshot: ConfigSnapshot) -> AppContext:
    memory = MemoryStore(config.paths.data_root)
    memory.initialize()
    llm = OllamaClient(config)

    registry = ToolRegistry()
    registry.register(Tool("local_search", "Search local facts and lessons", "safe", make_local_search_tool(config.paths.data_root)))
    registry.register(Tool("web_search", "Search web via Brave", "safe", make_web_search_tool(config)))
    registry.register(Tool("check_disk_space", "Check disk usage", "safe", check_disk_space))
    registry.register(Tool("check_docker_containers", "Docker containers (stub)", "safe", check_docker_containers_stub))

    context = AppContext(
        config=config,
        config_snapshot=snapshot,
        memory=memory,
        llm=llm,
        tools=registry,
        heartbeat=HeartbeatLoop(interval_seconds=config.heartbeat.interval_seconds, tick_fn=lambda: run_heartbeat_tick(memory)),
    )
    return context


def _build_messages_for_llm(memory: MemoryStore, session_id: str, user_message: str, recent_pairs: int, soul_rules: str) -> tuple[str, list[dict[str, str]]]:
    session = memory.load_or_create_session(session_id)
    profile = memory.load_profile()
    preferences = memory.load_preferences()
    recent = session.messages[-(recent_pairs * 2) :]
    system_prompt = (
        f"{soul_rules}\n\n"
        f"Profile: {profile}\n"
        f"Preferences: {preferences}\n"
        f"Session summary: {session.running_summary}\n"
    )
    messages = [{"role": m.role, "content": m.content} for m in recent]
    messages.append({"role": "user", "content": user_message})
    return system_prompt, messages


def handle_user_message(context: AppContext, session_id: str, user_message: str) -> str:
    """Handle user message and persist session updates."""
    context.memory.append_session_message(session_id=session_id, role="user", content=user_message)
    system_prompt, messages = _build_messages_for_llm(
        memory=context.memory,
        session_id=session_id,
        user_message=user_message,
        recent_pairs=context.config.session.recent_pairs,
        soul_rules=context.config.prompt.soul_rules,
    )
    answer = context.llm.generate(system_prompt=system_prompt, messages=messages, max_tokens=800)
    context.memory.append_session_message(session_id=session_id, role="assistant", content=answer)
    maint = context.config.session.maintenance
    context.memory.run_session_maintenance(
        mode=maint.mode,
        prune_after=maint.prune_after,
        max_entries=maint.max_entries,
        max_disk_bytes=maint.max_disk_bytes,
        high_water_ratio=maint.high_water_ratio,
        active_key=session_id,
    )
    return answer


def run_heartbeat_tick(memory: MemoryStore) -> None:
    """Run one heartbeat tick."""
    memory.append_episodic("Heartbeat tick")


def call_tool(context: AppContext, name: str, args: dict[str, Any]) -> dict:
    """Call registered tool by name."""
    tool = context.tools.get(name)
    if not tool:
        return {"ok": False, "tool": name, "data": None, "error": {"code": "not_found", "message": f"Unknown tool: {name}"}, "metadata": {}}
    return tool.fn(args)


def health_summary_quick(context: AppContext) -> str:
    """Quick health summary string."""
    config_ok = context.config_snapshot.valid
    data_ok = True
    llm_ok, llm_detail = context.llm.check_reachable(timeout_seconds=3)
    brave_ready = bool(context.config.tools.web_search.api_key or __import__("os").getenv("BLAIRE_BRAVE_API_KEY"))
    hb = context.heartbeat.status()
    return (
        f"config={'ok' if config_ok else 'invalid'}; "
        f"data={'ok' if data_ok else 'error'}; "
        f"llm={'ok' if llm_ok else f'fail({llm_detail})'}; "
        f"brave={'ready' if brave_ready else 'missing_key'}; "
        f"heartbeat={'running' if hb.running else 'stopped'}"
    )


def diagnostics(context: AppContext, deep: bool = False) -> dict[str, Any]:
    """Detailed diagnostics dictionary."""
    llm_timeout = 10 if deep else 3
    llm_ok, llm_detail = context.llm.check_reachable(timeout_seconds=llm_timeout)
    brave_key = bool(context.config.tools.web_search.api_key or __import__("os").getenv("BLAIRE_BRAVE_API_KEY"))
    lock_scan = {"status": "not_run"}
    brave_probe: dict[str, Any] = {"checked": False}
    if deep:
        scanned = clean_stale_locks(context.config.paths.data_root)
        lock_scan = {"status": "ok", **asdict(scanned)}
        if brave_key:
            import urllib.request

            url = "https://api.search.brave.com/res/v1/web/search?q=blaire+health&count=1"
            request = urllib.request.Request(
                url=url,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": context.config.tools.web_search.api_key
                    or __import__("os").getenv("BLAIRE_BRAVE_API_KEY", ""),
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                    brave_probe = {"checked": True, "ok": 200 <= response.status < 300, "status": response.status}
            except Exception as exc:  # noqa: BLE001
                brave_probe = {"checked": True, "ok": False, "error": str(exc)}
        else:
            brave_probe = {"checked": True, "ok": False, "error": "missing_key"}
    return {
        "config_valid": context.config_snapshot.valid,
        "config_issues": context.config_snapshot.issues,
        "data_root": context.config.paths.data_root,
        "llm": {"ok": llm_ok, "detail": llm_detail, "timeout_seconds": llm_timeout},
        "brave": {"ready": brave_key},
        "brave_probe": brave_probe,
        "heartbeat": asdict(context.heartbeat.status()),
        "deep": deep,
        "locks": lock_scan,
    }
