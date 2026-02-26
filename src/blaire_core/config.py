"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppSection:
    env: str


@dataclass(slots=True)
class PathsSection:
    data_root: str
    log_dir: str


@dataclass(slots=True)
class LLMSection:
    base_url: str
    model: str
    timeout_seconds: int
    temperature: float = 0.75
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_ctx: int = 8192


@dataclass(slots=True)
class HeartbeatSection:
    interval_seconds: int


@dataclass(slots=True)
class WebSearchSection:
    api_key: str
    timeout_seconds: int
    cache_ttl_minutes: int
    result_count: int
    safesearch: str
    auto_use: bool
    auto_count: int


@dataclass(slots=True)
class ToolPlannerSection:
    enabled: bool
    max_calls_per_turn: int
    confidence_threshold: float


@dataclass(slots=True)
class SSHHostSection:
    host: str
    user: str
    port: int
    key_path: str


@dataclass(slots=True)
class SSHSection:
    hosts: dict[str, SSHHostSection]
    connect_timeout_seconds: int
    command_allowlist: list[str]


@dataclass(slots=True)
class IntegrationServiceSection:
    base_url: str
    api_key: str
    username: str
    password: str
    collection: str


@dataclass(slots=True)
class ToolsIntegrationsSection:
    obsidian: IntegrationServiceSection
    qdrant: IntegrationServiceSection
    whisper: IntegrationServiceSection
    chatterbox: IntegrationServiceSection
    sonarr: IntegrationServiceSection
    radarr: IntegrationServiceSection
    qbittorrent: IntegrationServiceSection
    uptime_kuma: IntegrationServiceSection


@dataclass(slots=True)
class ToolApprovalsSection:
    token_ttl_seconds: int
    max_pending: int
    require_confirmation: bool


@dataclass(slots=True)
class ToolsSection:
    web_search: WebSearchSection
    planner: ToolPlannerSection
    ssh: SSHSection = field(
        default_factory=lambda: SSHSection(
            hosts={},
            connect_timeout_seconds=8,
            command_allowlist=[],
        )
    )
    integrations: ToolsIntegrationsSection = field(
        default_factory=lambda: ToolsIntegrationsSection(
            obsidian=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            qdrant=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            whisper=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            chatterbox=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            sonarr=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            radarr=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            qbittorrent=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            uptime_kuma=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
        )
    )
    approvals: ToolApprovalsSection = field(
        default_factory=lambda: ToolApprovalsSection(
            token_ttl_seconds=180,
            max_pending=50,
            require_confirmation=True,
        )
    )


@dataclass(slots=True)
class PromptSection:
    soul_rules: str


@dataclass(slots=True)
class SessionMaintenanceSection:
    mode: str
    prune_after: str
    max_entries: int
    max_disk_bytes: int | None
    high_water_ratio: float


@dataclass(slots=True)
class SessionSection:
    recent_pairs: int
    maintenance: SessionMaintenanceSection


@dataclass(slots=True)
class LoggingSection:
    level: str


@dataclass(slots=True)
class TelegramSection:
    enabled: bool
    bot_token: str | None
    chat_id: str | None
    polling_enabled: bool
    polling_timeout_seconds: int


@dataclass(slots=True)
class AppConfig:
    app: AppSection
    paths: PathsSection
    llm: LLMSection
    heartbeat: HeartbeatSection
    tools: ToolsSection
    prompt: PromptSection
    session: SessionSection
    logging: LoggingSection
    telegram: TelegramSection = field(
        default_factory=lambda: TelegramSection(
            enabled=False,
            bot_token=None,
            chat_id=None,
            polling_enabled=False,
            polling_timeout_seconds=20,
        )
    )


@dataclass(slots=True)
class ConfigSnapshot:
    path: str
    exists: bool
    valid: bool
    issues: list[str]
    warnings: list[str]
    effective_config: AppConfig | None
    effective_raw: dict[str, Any] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path(env: str) -> Path:
    return _repo_root() / "config" / f"{env}.json"


def _local_config_path() -> Path:
    return _repo_root() / "config" / "local.json"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _parse_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in dotted.split(".") if p]
    node: dict[str, Any] = target
    for part in parts[:-1]:
        current = node.get(part)
        if not isinstance(current, dict):
            current = {}
            node[part] = current
        node = current
    if parts:
        node[parts[-1]] = value


def _env_overrides() -> dict[str, Any]:
    mapping = {
        "BLAIRE_LLM_BASE_URL": "llm.base_url",
        "BLAIRE_LLM_MODEL": "llm.model",
        "BLAIRE_LLM_TEMPERATURE": "llm.temperature",
        "BLAIRE_LLM_TOP_P": "llm.top_p",
        "BLAIRE_LLM_REPEAT_PENALTY": "llm.repeat_penalty",
        "BLAIRE_LLM_NUM_CTX": "llm.num_ctx",
        "BLAIRE_DATA_PATH": "paths.data_root",
        "BLAIRE_HEARTBEAT_INTERVAL": "heartbeat.interval_seconds",
        "BLAIRE_BRAVE_API_KEY": "tools.web_search.api_key",
        "BLAIRE_OBSIDIAN_BASE_URL": "tools.integrations.obsidian.base_url",
        "BLAIRE_OBSIDIAN_API_KEY": "tools.integrations.obsidian.api_key",
        "BLAIRE_QDRANT_BASE_URL": "tools.integrations.qdrant.base_url",
        "BLAIRE_QDRANT_API_KEY": "tools.integrations.qdrant.api_key",
        "BLAIRE_QDRANT_COLLECTION": "tools.integrations.qdrant.collection",
        "BLAIRE_WHISPER_BASE_URL": "tools.integrations.whisper.base_url",
        "BLAIRE_CHATTERBOX_BASE_URL": "tools.integrations.chatterbox.base_url",
        "BLAIRE_SONARR_BASE_URL": "tools.integrations.sonarr.base_url",
        "BLAIRE_SONARR_API_KEY": "tools.integrations.sonarr.api_key",
        "BLAIRE_RADARR_BASE_URL": "tools.integrations.radarr.base_url",
        "BLAIRE_RADARR_API_KEY": "tools.integrations.radarr.api_key",
        "BLAIRE_QBITTORRENT_BASE_URL": "tools.integrations.qbittorrent.base_url",
        "BLAIRE_QBITTORRENT_USERNAME": "tools.integrations.qbittorrent.username",
        "BLAIRE_QBITTORRENT_PASSWORD": "tools.integrations.qbittorrent.password",
        "BLAIRE_UPTIME_KUMA_BASE_URL": "tools.integrations.uptime_kuma.base_url",
        "BLAIRE_UPTIME_KUMA_API_KEY": "tools.integrations.uptime_kuma.api_key",
        "BLAIRE_LOG_LEVEL": "logging.level",
        "BLAIRE_TELEGRAM_ENABLED": "telegram.enabled",
        "BLAIRE_TELEGRAM_BOT_TOKEN": "telegram.bot_token",
        "BLAIRE_TELEGRAM_CHAT_ID": "telegram.chat_id",
        "BLAIRE_TELEGRAM_POLLING_ENABLED": "telegram.polling_enabled",
        "BLAIRE_TELEGRAM_POLLING_TIMEOUT_SECONDS": "telegram.polling_timeout_seconds",
    }
    out: dict[str, Any] = {}
    for env_name, cfg_path in mapping.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        _set_path(out, cfg_path, _parse_override_value(raw))
    return out


def _validate(raw: dict[str, Any]) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    top_required = {"app", "paths", "llm", "heartbeat", "tools", "prompt", "session", "logging"}
    top_optional = {"telegram"}
    top_unknown = set(raw.keys()) - top_required - top_optional
    if top_unknown:
        issues.append(f"Unknown top-level keys: {', '.join(sorted(top_unknown))}")
    for key in sorted(top_required):
        if key not in raw:
            issues.append(f"Missing required section: {key}")

    llm_model = str(raw.get("llm", {}).get("model", "")).strip() if isinstance(raw.get("llm"), dict) else ""
    if not llm_model:
        issues.append("llm.model is required and cannot be empty")
    if isinstance(raw.get("llm"), dict):
        llm = raw["llm"]
        if not isinstance(llm.get("timeout_seconds"), int):
            issues.append("llm.timeout_seconds must be an integer")
        for key in ("temperature", "top_p", "repeat_penalty"):
            value = llm.get(key)
            if not isinstance(value, (int, float)):
                issues.append(f"llm.{key} must be a number")
        if not isinstance(llm.get("num_ctx"), int):
            issues.append("llm.num_ctx must be an integer")

    hb = raw.get("heartbeat", {})
    if isinstance(hb, dict):
        interval = hb.get("interval_seconds")
        if not isinstance(interval, int):
            issues.append("heartbeat.interval_seconds must be an integer")
    else:
        issues.append("heartbeat must be an object")

    tools = raw.get("tools", {})
    if isinstance(tools, dict):
        ws = tools.get("web_search", {})
        if not isinstance(ws, dict):
            issues.append("tools.web_search must be an object")
        else:
            safesearch = str(ws.get("safesearch", "off")).lower()
            if safesearch not in {"off", "moderate", "strict"}:
                issues.append("tools.web_search.safesearch must be off|moderate|strict")
            if not isinstance(ws.get("auto_use"), bool):
                issues.append("tools.web_search.auto_use must be a boolean")
            auto_count = ws.get("auto_count")
            if not isinstance(auto_count, int) or auto_count < 1 or auto_count > 10:
                issues.append("tools.web_search.auto_count must be an integer between 1 and 10")
        planner = tools.get("planner", {})
        if not isinstance(planner, dict):
            issues.append("tools.planner must be an object")
        else:
            if not isinstance(planner.get("enabled"), bool):
                issues.append("tools.planner.enabled must be a boolean")
            max_calls_per_turn = planner.get("max_calls_per_turn")
            if not isinstance(max_calls_per_turn, int) or max_calls_per_turn < 1 or max_calls_per_turn > 5:
                issues.append("tools.planner.max_calls_per_turn must be an integer between 1 and 5")
            confidence_threshold = planner.get("confidence_threshold")
            if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
                issues.append("tools.planner.confidence_threshold must be a number between 0 and 1")
        ssh = tools.get("ssh", {})
        if ssh and not isinstance(ssh, dict):
            issues.append("tools.ssh must be an object")
        approvals = tools.get("approvals", {})
        if approvals and not isinstance(approvals, dict):
            issues.append("tools.approvals must be an object")
        integrations = tools.get("integrations", {})
        if integrations and not isinstance(integrations, dict):
            issues.append("tools.integrations must be an object")
    else:
        issues.append("tools must be an object")

    session = raw.get("session", {})
    if isinstance(session, dict):
        maint = session.get("maintenance", {})
        if isinstance(maint, dict):
            mode = maint.get("mode")
            if mode not in {"warn", "enforce"}:
                issues.append("session.maintenance.mode must be warn|enforce")
        else:
            issues.append("session.maintenance must be an object")
    else:
        issues.append("session must be an object")

    telegram = raw.get("telegram", {})
    if isinstance(telegram, dict):
        enabled = bool(telegram.get("enabled", False))
        polling_enabled = bool(telegram.get("polling_enabled", False))
        polling_timeout_seconds = telegram.get("polling_timeout_seconds", 20)
        if not isinstance(polling_timeout_seconds, int) or polling_timeout_seconds < 1 or polling_timeout_seconds > 60:
            issues.append("telegram.polling_timeout_seconds must be an integer between 1 and 60")
        if enabled:
            token = str(telegram.get("bot_token", "")).strip()
            chat_id = str(telegram.get("chat_id", "")).strip()
            if not token or not chat_id:
                warnings.append(
                    "telegram is enabled but bot_token/chat_id is missing; Telegram notifications will be disabled"
                )
        if polling_enabled and not enabled:
            warnings.append("telegram polling enabled while telegram is disabled; polling will be disabled")
    else:
        issues.append("telegram must be an object when provided")

    return issues, warnings


def _to_config(raw: dict[str, Any]) -> AppConfig:
    ws = raw["tools"]["web_search"]
    planner = raw["tools"]["planner"]
    ssh_raw = raw["tools"].get("ssh", {}) if isinstance(raw["tools"].get("ssh"), dict) else {}
    approvals_raw = raw["tools"].get("approvals", {}) if isinstance(raw["tools"].get("approvals"), dict) else {}
    integrations_raw = raw["tools"].get("integrations", {}) if isinstance(raw["tools"].get("integrations"), dict) else {}
    maint = raw["session"]["maintenance"]
    telegram_raw = raw.get("telegram", {}) if isinstance(raw.get("telegram", {}), dict) else {}
    telegram_enabled = bool(telegram_raw.get("enabled", False))
    telegram_bot_token = str(telegram_raw.get("bot_token", "")).strip() or None
    telegram_chat_id = str(telegram_raw.get("chat_id", "")).strip() or None
    telegram_polling_enabled = bool(telegram_raw.get("polling_enabled", False))
    telegram_polling_timeout_seconds = int(telegram_raw.get("polling_timeout_seconds", 20))
    if telegram_enabled and (telegram_bot_token is None or telegram_chat_id is None):
        telegram_enabled = False
        telegram_polling_enabled = False
    if not telegram_enabled:
        telegram_polling_enabled = False

    def _service(name: str) -> IntegrationServiceSection:
        row = integrations_raw.get(name, {}) if isinstance(integrations_raw.get(name), dict) else {}
        return IntegrationServiceSection(
            base_url=str(row.get("base_url", "")),
            api_key=str(row.get("api_key", "")),
            username=str(row.get("username", "")),
            password=str(row.get("password", "")),
            collection=str(row.get("collection", "")),
        )

    ssh_hosts: dict[str, SSHHostSection] = {}
    for alias, row in (ssh_raw.get("hosts", {}) if isinstance(ssh_raw.get("hosts"), dict) else {}).items():
        if not isinstance(row, dict):
            continue
        ssh_hosts[str(alias)] = SSHHostSection(
            host=str(row.get("host", "")),
            user=str(row.get("user", "")),
            port=int(row.get("port", 22)),
            key_path=str(row.get("key_path", "")),
        )

    return AppConfig(
        app=AppSection(env=str(raw["app"]["env"])),
        paths=PathsSection(data_root=str(raw["paths"]["data_root"]), log_dir=str(raw["paths"]["log_dir"])),
        llm=LLMSection(
            base_url=str(raw["llm"]["base_url"]),
            model=str(raw["llm"]["model"]),
            timeout_seconds=int(raw["llm"]["timeout_seconds"]),
            temperature=float(raw["llm"].get("temperature", 0.75)),
            top_p=float(raw["llm"].get("top_p", 0.9)),
            repeat_penalty=float(raw["llm"].get("repeat_penalty", 1.1)),
            num_ctx=int(raw["llm"].get("num_ctx", 8192)),
        ),
        heartbeat=HeartbeatSection(interval_seconds=int(raw["heartbeat"]["interval_seconds"])),
        tools=ToolsSection(
            web_search=WebSearchSection(
                api_key=str(ws.get("api_key", "")),
                timeout_seconds=int(ws["timeout_seconds"]),
                cache_ttl_minutes=int(ws["cache_ttl_minutes"]),
                result_count=int(ws["result_count"]),
                safesearch=str(ws["safesearch"]),
                auto_use=bool(ws.get("auto_use", True)),
                auto_count=int(ws.get("auto_count", 3)),
            ),
            planner=ToolPlannerSection(
                enabled=bool(planner.get("enabled", True)),
                max_calls_per_turn=int(planner.get("max_calls_per_turn", 2)),
                confidence_threshold=float(planner.get("confidence_threshold", 0.55)),
            ),
            ssh=SSHSection(
                hosts=ssh_hosts,
                connect_timeout_seconds=int(ssh_raw.get("connect_timeout_seconds", 8)),
                command_allowlist=[str(v) for v in ssh_raw.get("command_allowlist", []) if isinstance(v, str)],
            ),
            integrations=ToolsIntegrationsSection(
                obsidian=_service("obsidian"),
                qdrant=_service("qdrant"),
                whisper=_service("whisper"),
                chatterbox=_service("chatterbox"),
                sonarr=_service("sonarr"),
                radarr=_service("radarr"),
                qbittorrent=_service("qbittorrent"),
                uptime_kuma=_service("uptime_kuma"),
            ),
            approvals=ToolApprovalsSection(
                token_ttl_seconds=int(approvals_raw.get("token_ttl_seconds", 180)),
                max_pending=int(approvals_raw.get("max_pending", 50)),
                require_confirmation=bool(approvals_raw.get("require_confirmation", True)),
            ),
        ),
        prompt=PromptSection(soul_rules=str(raw["prompt"]["soul_rules"])),
        session=SessionSection(
            recent_pairs=int(raw["session"]["recent_pairs"]),
            maintenance=SessionMaintenanceSection(
                mode=str(maint["mode"]),
                prune_after=str(maint["prune_after"]),
                max_entries=int(maint["max_entries"]),
                max_disk_bytes=maint.get("max_disk_bytes"),
                high_water_ratio=float(maint["high_water_ratio"]),
            ),
        ),
        logging=LoggingSection(level=str(raw["logging"]["level"])),
        telegram=TelegramSection(
            enabled=telegram_enabled,
            bot_token=telegram_bot_token,
            chat_id=telegram_chat_id,
            polling_enabled=telegram_polling_enabled,
            polling_timeout_seconds=telegram_polling_timeout_seconds,
        ),
    )


def _bootstrap_config(env: str) -> AppConfig:
    """Minimal runtime-safe config used when file config is invalid."""
    return AppConfig(
        app=AppSection(env=env),
        paths=PathsSection(data_root="./data", log_dir="data/logs"),
        llm=LLMSection(
            base_url="http://192.168.0.10:11434",
            model="bootstrap-fallback",
            timeout_seconds=30,
            temperature=0.75,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=8192,
        ),
        heartbeat=HeartbeatSection(interval_seconds=0),
        tools=ToolsSection(
            web_search=WebSearchSection(
                api_key="",
                timeout_seconds=10,
                cache_ttl_minutes=15,
                result_count=10,
                safesearch="off",
                auto_use=True,
                auto_count=3,
            ),
            planner=ToolPlannerSection(
                enabled=True,
                max_calls_per_turn=2,
                confidence_threshold=0.55,
            ),
            ssh=SSHSection(
                hosts={},
                connect_timeout_seconds=8,
                command_allowlist=[],
            ),
            integrations=ToolsIntegrationsSection(
                obsidian=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
                qdrant=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
                whisper=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
                chatterbox=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
                sonarr=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
                radarr=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
                qbittorrent=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
                uptime_kuma=IntegrationServiceSection(base_url="", api_key="", username="", password="", collection=""),
            ),
            approvals=ToolApprovalsSection(
                token_ttl_seconds=180,
                max_pending=50,
                require_confirmation=True,
            ),
        ),
        prompt=PromptSection(soul_rules="You are BLAIRE Core. Be concise, safe, and practical."),
        session=SessionSection(
            recent_pairs=6,
            maintenance=SessionMaintenanceSection(
                mode="warn",
                prune_after="30d",
                max_entries=500,
                max_disk_bytes=None,
                high_water_ratio=0.8,
            ),
        ),
        logging=LoggingSection(level="info"),
        telegram=TelegramSection(
            enabled=False,
            bot_token=None,
            chat_id=None,
            polling_enabled=False,
            polling_timeout_seconds=20,
        ),
    )


def read_config_snapshot(env: str, cli_overrides: dict[str, str] | None = None) -> ConfigSnapshot:
    """Read config file and return validity snapshot."""
    path = _default_config_path(env)
    issues: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return ConfigSnapshot(
            path=str(path),
            exists=False,
            valid=False,
            issues=[f"Config file does not exist: {path}"],
            warnings=[],
            effective_config=None,
            effective_raw=None,
        )

    try:
        file_raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return ConfigSnapshot(
            path=str(path),
            exists=True,
            valid=False,
            issues=[f"Failed to parse config JSON: {exc}"],
            warnings=[],
            effective_config=None,
            effective_raw=None,
        )
    if not isinstance(file_raw, dict):
        return ConfigSnapshot(
            path=str(path),
            exists=True,
            valid=False,
            issues=["Top-level config must be an object"],
            warnings=[],
            effective_config=None,
            effective_raw=None,
        )

    merged = dict(file_raw)
    local_path = _local_config_path()
    if local_path.exists():
        try:
            local_raw = json.loads(local_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return ConfigSnapshot(
                path=str(path),
                exists=True,
                valid=False,
                issues=[f"Failed to parse local config JSON ({local_path}): {exc}"],
                warnings=[],
                effective_config=None,
                effective_raw=None,
            )
        if not isinstance(local_raw, dict):
            return ConfigSnapshot(
                path=str(path),
                exists=True,
                valid=False,
                issues=[f"Local config must be a JSON object: {local_path}"],
                warnings=[],
                effective_config=None,
                effective_raw=None,
            )
        merged = _deep_update(merged, local_raw)
        warnings.append(f"Applied local config overrides from {local_path}")
    merged = _deep_update(merged, _env_overrides())
    if cli_overrides:
        cli_tree: dict[str, Any] = {}
        for key, value in cli_overrides.items():
            _set_path(cli_tree, key, _parse_override_value(value))
        merged = _deep_update(merged, cli_tree)

    issues, warnings = _validate(merged)
    if issues:
        return ConfigSnapshot(
            path=str(path),
            exists=True,
            valid=False,
            issues=issues,
            warnings=warnings,
            effective_config=None,
            effective_raw=None,
        )
    return ConfigSnapshot(
        path=str(path),
        exists=True,
        valid=True,
        issues=[],
        warnings=warnings,
        effective_config=_to_config(merged),
        effective_raw=merged,
    )


def ensure_runtime_config(snapshot: ConfigSnapshot, env: str) -> AppConfig:
    """Return valid runtime config; fallback to bootstrap config when snapshot invalid."""
    if snapshot.effective_config is not None:
        return snapshot.effective_config
    return _bootstrap_config(env)
