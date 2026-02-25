"""Core orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import re
import os
import time

from blaire_core.config import AppConfig, ConfigSnapshot
from blaire_core.heartbeat.jobs import run_heartbeat_jobs
from blaire_core.heartbeat.loop import HeartbeatLoop
from blaire_core.learning.routine import apply_learning_updates
from blaire_core.learning.soul_growth import apply_soul_growth_updates
from blaire_core.llm.client import OllamaClient
from blaire_core.memory.store import MemoryStore, clean_stale_locks
from blaire_core.notifications import notify_user
from blaire_core.prompting.composer import build_system_prompt
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
    tool_runtime: dict[str, ToolRuntimeStats]


@dataclass(slots=True)
class ToolRuntimeStats:
    selection_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    fallback_count: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float | None = None
    last_error_code: str | None = None
    last_called_at: float | None = None
    cooldown_until: float = 0.0
    call_timestamps: deque[float] | None = None

    def __post_init__(self) -> None:
        if self.call_timestamps is None:
            self.call_timestamps = deque()

    @property
    def success_rate(self) -> float:
        if self.selection_count == 0:
            return 0.0
        return self.success_count / self.selection_count

    @property
    def failure_rate(self) -> float:
        if self.selection_count == 0:
            return 0.0
        return self.failure_count / self.selection_count

    @property
    def average_latency_ms(self) -> float:
        if self.selection_count == 0:
            return 0.0
        return self.total_latency_ms / self.selection_count


def _tool_stats(context: AppContext, tool_name: str) -> ToolRuntimeStats:
    if tool_name not in context.tool_runtime:
        context.tool_runtime[tool_name] = ToolRuntimeStats()
    return context.tool_runtime[tool_name]


def _log_tool_telemetry(context: AppContext, tool_name: str, stats: ToolRuntimeStats, status: str, payload: dict[str, Any]) -> None:
    context.memory.log_event(
        event_type="tool_telemetry",
        payload={
            "tool": tool_name,
            "status": status,
            "selection_count": stats.selection_count,
            "success_count": stats.success_count,
            "failure_count": stats.failure_count,
            "fallback_count": stats.fallback_count,
            "success_rate": round(stats.success_rate, 4),
            "failure_rate": round(stats.failure_rate, 4),
            "average_latency_ms": round(stats.average_latency_ms, 2),
            "last_latency_ms": None if stats.last_latency_ms is None else round(stats.last_latency_ms, 2),
            **payload,
        },
        session_id=None,
    )


_AUTO_WEB_SEARCH_PATTERNS = [
    r"\b(latest|most recent|today|news|breaking)\b",
    r"\b(current|currently|as of)\b",
    r"\b(price|stock|market|weather|score)\b",
    r"\b(version|release|changelog)\b",
    r"\b(search|look up|lookup)\b.{0,40}\b(web|internet)\b",
]


def _should_auto_web_search(user_message: str) -> bool:
    text = user_message.strip().lower()
    if not text:
        return False
    if any(re.search(pattern, text) for pattern in _AUTO_WEB_SEARCH_PATTERNS):
        return True
    return text.startswith(("who is", "what is", "when is", "where is", "how to")) and text.endswith("?")


def _capability_guard_message() -> str:
    return (
        "Capability Guard (non-negotiable):\n"
        "- You are BLAIRE running inside a local runtime with persistent memory (session, episodic, long-term, structured DB).\n"
        "- You have tool-mediated web search capability when configured; do not claim static cutoff/no-internet by default.\n"
        "- If a tool/config issue prevents a capability, state that specific failure and offer next steps.\n"
        "- Never claim 'I cannot retain context between sessions' in this runtime."
    )


_CAPABILITY_DRIFT_PATTERNS = [
    r"\bi don't have internet access\b",
    r"\bi do not have internet access\b",
    r"\bi don't have memory beyond this session\b",
    r"\bcan't retain context between sessions\b",
    r"\bcannot retain context between sessions\b",
    r"\bknowledge is static\b",
    r"\b2023 cutoff\b",
    r"\bmemory is disabled\b",
]


def _has_capability_drift(answer: str) -> bool:
    lowered = answer.strip().lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered) for pattern in _CAPABILITY_DRIFT_PATTERNS)


def _is_memory_recall_prompt(user_message: str) -> bool:
    lowered = user_message.strip().lower()
    return bool(
        re.search(r"\bwhat is my name\b", lowered)
        or re.search(r"\bwhat is my long[- ]term goal\b", lowered)
        or re.search(r"\bstate my name and goal\b", lowered)
    )


def _memory_recall_fallback(memory: MemoryStore, user_message: str) -> str:
    profile = memory.load_profile()
    name = str(profile.get("name", "") or "").strip()
    goals = [str(v).strip() for v in profile.get("long_term_goals", []) if isinstance(v, str) and str(v).strip()]
    lowered = user_message.strip().lower()
    if "state my name and goal" in lowered:
        if name and goals:
            return f"Your name is {name} and your long-term goal is {goals[-1]}."
        if name:
            return f"Your name is {name}. I do not yet have a long-term goal stored."
        if goals:
            return f"I do not yet have your name stored. Your long-term goal is {goals[-1]}."
        return "I do not yet have your name or long-term goal stored."
    if "what is my name" in lowered:
        if name:
            return f"Your name is {name}."
        return "I do not yet have your name stored."
    if "what is my long-term goal" in lowered or "what is my long term goal" in lowered:
        if goals:
            return f"Your long-term goal is {goals[-1]}."
        return "I do not yet have a long-term goal stored."
    return ""


def _memory_recall_answer_looks_invalid(answer: str) -> bool:
    lowered = answer.strip().lower()
    return bool(
        re.search(r"\b(memory is disabled|cannot recall|can't recall|do not recall|no prior interactions)\b", lowered)
    )


def _capability_safe_fallback(user_message: str, web_attempted: bool) -> str:
    lowered = user_message.strip().lower()
    asks_web = bool(re.search(r"\b(search|look up|lookup)\b.{0,40}\b(web|internet)\b", lowered)) or _should_auto_web_search(
        user_message
    )
    if asks_web:
        if web_attempted:
            return (
                "Yes. I have persistent memory in this runtime and I can use the web-search tool here. "
                "Share the exact query and I will run it now."
            )
        return (
            "Yes. I have persistent memory in this runtime and can run web search when configured. "
            "Give me the exact query and I will search it now."
        )
    return (
        "I have persistent memory in this runtime and I can use configured tools (including web search). "
        "Tell me the next task and I will execute it."
    )


def _build_web_context(web_result: dict[str, Any]) -> str:
    if not web_result.get("ok"):
        error = web_result.get("error", {})
        return f"Web search unavailable: {error.get('code', 'unknown')} - {error.get('message', '')}"
    data = web_result.get("data", {})
    query = data.get("query", "")
    provider = data.get("provider", "brave")
    rows: list[str] = [f"Web search context ({provider}) for query: {query}"]
    for item in data.get("results", [])[:3]:
        rows.append(f"- {item.get('title', '')} | {item.get('url', '')}")
        snippet = str(item.get("snippet", "")).replace("\n", " ")
        rows.append(f"  snippet: {snippet[:280]}")
    return "\n".join(rows)


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
        heartbeat=HeartbeatLoop(
            interval_seconds=config.heartbeat.interval_seconds,
            tick_fn=lambda: run_heartbeat_tick(memory, config, llm),
        ),
        tool_runtime={name: ToolRuntimeStats() for name in registry.names()},
    )
    return context


def _build_messages_for_llm(memory: MemoryStore, session_id: str, user_message: str, recent_pairs: int, soul_rules: str) -> tuple[str, list[dict[str, str]]]:
    session = memory.load_or_create_session(session_id)
    recent = session.messages[-(recent_pairs * 2) :]
    recent_lines = [f"{m.role}: {m.content}" for m in recent[-6:]]
    memory_query = "\n".join([*recent_lines, f"user: {user_message}"])
    include_patterns = bool(re.search(r"\b(pattern|behavio[u]?r|audit|introspect)\b", user_message, re.IGNORECASE))
    system_prompt = build_system_prompt(
        memory=memory,
        soul_rules=soul_rules,
        session_id=session_id,
        memory_query=memory_query,
        include_pattern_context=include_patterns,
    )
    system_prompt = f"{system_prompt}\n\n# Session Summary\n{session.running_summary or '(none)'}"
    messages = [{"role": "system", "content": _capability_guard_message()}]
    messages.extend({"role": m.role, "content": m.content} for m in recent)
    messages.append({"role": "user", "content": user_message})
    return system_prompt, messages


def handle_user_message(context: AppContext, session_id: str, user_message: str) -> str:
    """Handle user message and persist session updates."""
    context.memory.append_session_message(session_id=session_id, role="user", content=user_message)
    context.memory.log_event(
        event_type="user_message",
        session_id=session_id,
        payload={"content": user_message},
    )
    system_prompt, messages = _build_messages_for_llm(
        memory=context.memory,
        session_id=session_id,
        user_message=user_message,
        recent_pairs=context.config.session.recent_pairs,
        soul_rules=context.config.prompt.soul_rules,
    )
    web_attempted = False
    if context.config.tools.web_search.auto_use and _should_auto_web_search(user_message):
        tool = context.tools.get("web_search")
        if tool:
            web_result = tool.fn({"query": user_message, "count": context.config.tools.web_search.auto_count})
            messages.insert(0, {"role": "system", "content": _build_web_context(web_result)})
            web_attempted = True

    answer = context.llm.generate(system_prompt=system_prompt, messages=messages, max_tokens=800)
    if _has_capability_drift(answer):
        context.memory.log_event(
            event_type="capability_drift_detected",
            session_id=session_id,
            payload={"assistant_answer": answer[:1000]},
        )
        correction = (
            "Capability correction:\n"
            "- In this runtime you do have persistent memory and tool-mediated web search.\n"
            "- Do not deny these capabilities. If a specific tool call failed, state that failure precisely."
        )
        corrected_messages = [{"role": "system", "content": correction}, *messages]
        answer = context.llm.generate(system_prompt=system_prompt, messages=corrected_messages, max_tokens=800)
        if _has_capability_drift(answer):
            if _is_memory_recall_prompt(user_message):
                recall_fallback = _memory_recall_fallback(context.memory, user_message)
                if recall_fallback:
                    answer = recall_fallback
                else:
                    answer = _capability_safe_fallback(user_message, web_attempted=web_attempted)
            else:
                answer = _capability_safe_fallback(user_message, web_attempted=web_attempted)
    if _is_memory_recall_prompt(user_message) and _memory_recall_answer_looks_invalid(answer):
        fallback = _memory_recall_fallback(context.memory, user_message)
        if fallback:
            answer = fallback
    context.memory.append_session_message(session_id=session_id, role="assistant", content=answer)
    context.memory.log_event(
        event_type="assistant_message",
        session_id=session_id,
        payload={"content": answer},
    )
    notify_user(context.config, answer, level="info", via_telegram=True)
    learning = apply_learning_updates(context.memory, user_message=user_message, assistant_message=answer)
    soul_growth = apply_soul_growth_updates(context.memory, user_message=user_message, assistant_message=answer)
    if learning["profile_updates"] or learning["preferences_updates"] or learning["facts_added"] or soul_growth["updated"]:
        context.memory.append_episodic(f"Learning updates: {learning}; soul_growth: {soul_growth}")
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


def run_heartbeat_tick(memory: MemoryStore, config: AppConfig | None = None, llm: OllamaClient | None = None) -> None:
    """Run one heartbeat tick."""
    if config is not None:
        run_heartbeat_jobs(config, llm_client=llm)
    memory.append_episodic("Heartbeat tick")
    memory.log_event(event_type="heartbeat_tick", payload={"ok": True}, session_id=None)
    if config is not None and os.getenv("BLAIRE_HEARTBEAT_NOTIFY", "").strip().lower() == "true":
        notify_user(config, "Heartbeat tick executed", level="info")


def call_tool(context: AppContext, name: str, args: dict[str, Any]) -> dict:
    """Call registered tool by name."""
    tool = context.tools.get(name)
    if not tool:
        context.memory.log_event(
            event_type="tool_telemetry",
            payload={"tool": name, "status": "not_found", "selection_count": 1, "success_count": 0, "failure_count": 1},
            session_id=None,
        )
        return {"ok": False, "tool": name, "data": None, "error": {"code": "not_found", "message": f"Unknown tool: {name}"}, "metadata": {}}
    stats = _tool_stats(context, name)
    stats.selection_count += 1
    now = time.monotonic()
    stats.last_called_at = now

    payload_size = len(json.dumps(args, ensure_ascii=False).encode("utf-8"))
    if tool.max_payload_bytes is not None and payload_size > tool.max_payload_bytes:
        stats.failure_count += 1
        stats.fallback_count += 1
        stats.last_error_code = "payload_too_large"
        _log_tool_telemetry(
            context,
            name,
            stats,
            status="blocked",
            payload={"error_code": "payload_too_large", "payload_bytes": payload_size, "max_payload_bytes": tool.max_payload_bytes},
        )
        return {
            "ok": False,
            "tool": name,
            "data": None,
            "error": {
                "code": "payload_too_large",
                "message": f"Payload size {payload_size} exceeds max {tool.max_payload_bytes} bytes",
            },
            "metadata": {"payload_bytes": payload_size, "max_payload_bytes": tool.max_payload_bytes},
        }

    if tool.calls_per_minute is not None:
        window_start = now - 60.0
        while stats.call_timestamps and stats.call_timestamps[0] < window_start:
            stats.call_timestamps.popleft()
        if len(stats.call_timestamps) >= tool.calls_per_minute:
            stats.failure_count += 1
            stats.fallback_count += 1
            stats.last_error_code = "rate_limited"
            _log_tool_telemetry(
                context,
                name,
                stats,
                status="blocked",
                payload={"error_code": "rate_limited", "calls_per_minute": tool.calls_per_minute},
            )
            return {
                "ok": False,
                "tool": name,
                "data": None,
                "error": {"code": "rate_limited", "message": f"Tool '{name}' exceeded calls_per_minute={tool.calls_per_minute}"},
                "metadata": {"calls_per_minute": tool.calls_per_minute},
            }

    if tool.cooldown_seconds is not None and now < stats.cooldown_until:
        retry_after = max(0.0, stats.cooldown_until - now)
        stats.failure_count += 1
        stats.fallback_count += 1
        stats.last_error_code = "cooldown_active"
        _log_tool_telemetry(
            context,
            name,
            stats,
            status="blocked",
            payload={"error_code": "cooldown_active", "retry_after_seconds": round(retry_after, 3)},
        )
        return {
            "ok": False,
            "tool": name,
            "data": None,
            "error": {"code": "cooldown_active", "message": f"Tool '{name}' is cooling down"},
            "metadata": {"retry_after_seconds": retry_after},
        }

    started = time.monotonic()
    try:
        if tool.timeout_seconds is not None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.fn, args)
                result = future.result(timeout=tool.timeout_seconds)
        else:
            result = tool.fn(args)
    except FutureTimeoutError:
        latency_ms = (time.monotonic() - started) * 1000.0
        stats.failure_count += 1
        stats.fallback_count += 1
        stats.total_latency_ms += latency_ms
        stats.last_latency_ms = latency_ms
        stats.last_error_code = "timeout"
        if tool.cooldown_seconds is not None:
            stats.cooldown_until = time.monotonic() + tool.cooldown_seconds
        _log_tool_telemetry(
            context,
            name,
            stats,
            status="failure",
            payload={"error_code": "timeout", "timeout_seconds": tool.timeout_seconds},
        )
        return {
            "ok": False,
            "tool": name,
            "data": None,
            "error": {"code": "timeout", "message": f"Tool '{name}' exceeded timeout of {tool.timeout_seconds} seconds"},
            "metadata": {"timeout_seconds": tool.timeout_seconds},
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - started) * 1000.0
        stats.failure_count += 1
        stats.fallback_count += 1
        stats.total_latency_ms += latency_ms
        stats.last_latency_ms = latency_ms
        stats.last_error_code = "execution_failed"
        if tool.cooldown_seconds is not None:
            stats.cooldown_until = time.monotonic() + tool.cooldown_seconds
        _log_tool_telemetry(
            context,
            name,
            stats,
            status="failure",
            payload={"error_code": "execution_failed", "error": str(exc)[:500]},
        )
        return {
            "ok": False,
            "tool": name,
            "data": None,
            "error": {"code": "execution_failed", "message": str(exc)},
            "metadata": {},
        }

    latency_ms = (time.monotonic() - started) * 1000.0
    stats.total_latency_ms += latency_ms
    stats.last_latency_ms = latency_ms
    if tool.calls_per_minute is not None:
        stats.call_timestamps.append(time.monotonic())
    if tool.cooldown_seconds is not None:
        stats.cooldown_until = time.monotonic() + tool.cooldown_seconds

    if isinstance(result, dict) and bool(result.get("ok", True)):
        stats.success_count += 1
        stats.last_error_code = None
        _log_tool_telemetry(context, name, stats, status="success", payload={"latency_ms": round(latency_ms, 2)})
        return result

    stats.failure_count += 1
    stats.fallback_count += 1
    stats.last_error_code = "tool_returned_failure"
    _log_tool_telemetry(context, name, stats, status="failure", payload={"latency_ms": round(latency_ms, 2), "error_code": "tool_returned_failure"})
    if isinstance(result, dict):
        result.setdefault("tool", name)
        result.setdefault("metadata", {})
        return result
    return {
        "ok": False,
        "tool": name,
        "data": None,
        "error": {"code": "invalid_response", "message": "Tool returned a non-dict response"},
        "metadata": {},
    }


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
    tool_trends: dict[str, Any] = {}
    now = time.monotonic()
    for tool_name in context.tools.names():
        tool = context.tools.get(tool_name)
        if tool is None:
            continue
        stats = _tool_stats(context, tool_name)
        tool_trends[tool_name] = {
            "limits": {
                "calls_per_minute": tool.calls_per_minute,
                "timeout_seconds": tool.timeout_seconds,
                "cooldown_seconds": tool.cooldown_seconds,
                "max_payload_bytes": tool.max_payload_bytes,
            },
            "usage": {
                "selection_count": stats.selection_count,
                "success_count": stats.success_count,
                "failure_count": stats.failure_count,
                "fallback_count": stats.fallback_count,
                "success_rate": round(stats.success_rate, 4),
                "failure_rate": round(stats.failure_rate, 4),
            },
            "latency": {
                "average_ms": round(stats.average_latency_ms, 2),
                "last_ms": None if stats.last_latency_ms is None else round(stats.last_latency_ms, 2),
            },
            "health": {
                "last_error_code": stats.last_error_code,
                "cooldown_active": now < stats.cooldown_until,
                "cooldown_remaining_seconds": max(0.0, round(stats.cooldown_until - now, 3)),
            },
        }
    return {
        "config_valid": context.config_snapshot.valid,
        "config_issues": context.config_snapshot.issues,
        "data_root": context.config.paths.data_root,
        "llm": {"ok": llm_ok, "detail": llm_detail, "timeout_seconds": llm_timeout},
        "brave": {"ready": brave_key},
        "brave_probe": brave_probe,
        "heartbeat": asdict(context.heartbeat.status()),
        "tools": tool_trends,
        "deep": deep,
        "locks": lock_scan,
    }
