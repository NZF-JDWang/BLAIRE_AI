"""Core orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import re
import os
import json

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
class ToolPlan:
    tool_name: str
    args: dict[str, Any]
    reason: str
    confidence: float
    requires_followup: bool = False


def plan_tool_calls(
    context: AppContext, session_id: str, user_message: str, recent_messages: list[dict[str, str]]
) -> list[ToolPlan]:
    _ = (session_id, recent_messages)
    lowered = user_message.strip().lower()
    if not lowered:
        return []

    plans: list[ToolPlan] = []
    tool_map = {tool.name: tool for tool in context.tools.all_tools()}

    def _add(name: str, args: dict[str, Any], reason: str, confidence: float, requires_followup: bool = False) -> None:
        tool = tool_map.get(name)
        if not tool:
            return
        grounded_reason = reason
        if tool.usage_hints:
            grounded_reason = f"{reason} Hint: {tool.usage_hints[0]}"
        plans.append(
            ToolPlan(
                tool_name=name,
                args=args,
                reason=grounded_reason,
                confidence=confidence,
                requires_followup=requires_followup,
            )
        )

    if context.config.tools.planner.enabled:
        if context.config.tools.web_search.auto_use and any(word in lowered for word in ("latest", "today", "news", "current", "web", "internet")):
            _add(
                "web_search",
                {"query": user_message, "count": context.config.tools.web_search.auto_count},
                "Message likely needs current/external web data.",
                0.9,
            )

        local_terms = ("remember", "from memory", "we discussed", "stored", "saved", "long-term", "long term")
        if any(term in lowered for term in local_terms):
            _add("local_search", {"query": user_message, "limit": 10}, "Message asks for recalled local knowledge.", 0.82)

        if any(term in lowered for term in ("disk", "storage", "free space", "filesystem usage")):
            _add("check_disk_space", {"path": "."}, "Message asks for system storage status.", 0.86, requires_followup=True)

        if any(term in lowered for term in ("docker", "containers", "container status")):
            _add("check_docker_containers", {}, "Message asks for container health/status.", 0.74, requires_followup=True)
    elif context.config.tools.web_search.auto_use and _should_auto_web_search(user_message):
        _add(
            "web_search",
            {"query": user_message, "count": context.config.tools.web_search.auto_count},
            "Fallback regex web-search trigger while planner is disabled.",
            0.7,
        )

    threshold = context.config.tools.planner.confidence_threshold
    selected: list[ToolPlan] = []
    for plan in sorted(plans, key=lambda row: row.confidence, reverse=True):
        if plan.confidence < threshold:
            continue
        if any(existing.tool_name == plan.tool_name for existing in selected):
            continue
        selected.append(plan)
        if len(selected) >= context.config.tools.planner.max_calls_per_turn:
            break
    return selected


@dataclass(slots=True)
class AppContext:
    config: AppConfig
    config_snapshot: ConfigSnapshot
    memory: MemoryStore
    llm: OllamaClient
    tools: ToolRegistry
    heartbeat: HeartbeatLoop


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
    registry.register(
        Tool(
            "local_search",
            "Search local facts and lessons",
            "safe",
            make_local_search_tool(config.paths.data_root),
            arg_schema={"query": "string (required)", "limit": "int 1-50"},
            usage_hints=["Finds stored local memory/facts", "Best for recall and prior notes"],
        )
    )
    registry.register(
        Tool(
            "web_search",
            "Search web via Brave",
            "safe",
            make_web_search_tool(config),
            arg_schema={"query": "string (required)", "count": "int 1-10", "freshness": "optional string"},
            usage_hints=["Use for current events or internet lookups", "Requires Brave API key"],
        )
    )
    registry.register(
        Tool(
            "check_disk_space",
            "Check disk usage",
            "safe",
            check_disk_space,
            arg_schema={"path": "string path (optional)"},
            usage_hints=["Use for storage capacity checks"],
        )
    )
    registry.register(
        Tool(
            "check_docker_containers",
            "Docker containers (stub)",
            "safe",
            check_docker_containers_stub,
            arg_schema={},
            usage_hints=["Use for docker/container diagnostics"],
        )
    )

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
    recent_messages = messages[-6:]
    tool_plans = plan_tool_calls(context, session_id, user_message, recent_messages)
    for plan in tool_plans:
        tool = context.tools.get(plan.tool_name)
        if not tool:
            continue
        result = tool.fn(plan.args)
        if plan.tool_name == "web_search":
            messages.insert(0, {"role": "system", "content": _build_web_context(result)})
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
    _persist_assistant_turn(context, session_id, user_message, answer)
    return answer


def _persist_assistant_turn(context: AppContext, session_id: str, user_message: str, answer: str) -> None:
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


def handle_user_message_with_tool(
    context: AppContext,
    session_id: str,
    user_message: str,
    tool_name: str,
    args: dict[str, Any],
    debug_mode: bool = False,
) -> str:
    """Handle a user message by executing a specific tool first."""
    context.memory.append_session_message(session_id=session_id, role="user", content=user_message)
    context.memory.log_event(
        event_type="user_message",
        session_id=session_id,
        payload={"content": user_message, "tool_routed": tool_name},
    )
    tool_result = call_tool(context, name=tool_name, args=args)

    if debug_mode:
        answer = json.dumps(tool_result, indent=2)
        _persist_assistant_turn(context, session_id, user_message, answer)
        return answer

    system_prompt, messages = _build_messages_for_llm(
        memory=context.memory,
        session_id=session_id,
        user_message=user_message,
        recent_pairs=context.config.session.recent_pairs,
        soul_rules=context.config.prompt.soul_rules,
    )
    tool_context = {
        "tool": tool_name,
        "ok": bool(tool_result.get("ok")),
        "data": tool_result.get("data"),
        "error": tool_result.get("error"),
        "metadata": tool_result.get("metadata", {}),
    }
    messages.insert(
        0,
        {
            "role": "system",
            "content": (
                "Tool execution result is available. Use it directly and answer in BLAIRE's conversational voice. "
                "Do not expose raw JSON unless the user explicitly asked for debug output.\n"
                f"{json.dumps(tool_context, ensure_ascii=False)}"
            ),
        },
    )
    answer = context.llm.generate(system_prompt=system_prompt, messages=messages, max_tokens=800)
    _persist_assistant_turn(context, session_id, user_message, answer)
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
