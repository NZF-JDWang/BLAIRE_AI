from __future__ import annotations

import json
import urllib.request

from blaire_core.config import (
    AppConfig,
    AppSection,
    CalendarSection,
    EmailSection,
    HeartbeatSection,
    HomeAssistantSection,
    LLMSection,
    LoggingSection,
    ObsidianSection,
    PathsSection,
    PromptSection,
    ServerHealthSection,
    SessionMaintenanceSection,
    SessionSection,
    ToolsSection,
    WebSearchSection,
)
from blaire_core.tools.builtin_tools import (
    make_calendar_summary_tool,
    make_check_server_health_tool,
    make_email_summary_tool,
    make_home_assistant_read_tool,
    make_obsidian_search_tool,
)


class _FakeResponse:
    def __init__(self, payload: dict | list) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)


def _config() -> AppConfig:
    return AppConfig(
        app=AppSection(env="dev"),
        paths=PathsSection(data_root="./data", log_dir="data/logs"),
        llm=LLMSection(base_url="http://localhost:11434", model="m", timeout_seconds=30),
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
            server_health=ServerHealthSection(
                endpoints=["https://health.local/node-1"], api_token="health-token", timeout_seconds=8, scope="read:health"
            ),
            home_assistant=HomeAssistantSection(
                base_url="https://ha.local", access_token="ha-token", timeout_seconds=8, scope="read:states"
            ),
            obsidian=ObsidianSection(
                base_url="https://obsidian.local", api_key="obsidian-token", vault="main", timeout_seconds=8, scope="read:notes"
            ),
            calendar=CalendarSection(
                base_url="https://calendar.local", api_token="calendar-token", timeout_seconds=8, scope="read:events"
            ),
            email=EmailSection(
                base_url="https://mail.local", api_token="mail-token", timeout_seconds=8, scope="read:inbox"
            ),
        ),
        prompt=PromptSection(soul_rules="x"),
        session=SessionSection(
            recent_pairs=6,
            maintenance=SessionMaintenanceSection(
                mode="warn", prune_after="30d", max_entries=500, max_disk_bytes=None, high_water_ratio=0.8
            ),
        ),
        logging=LoggingSection(level="info"),
    )


def test_read_only_tools_missing_credentials() -> None:
    cfg = _config()
    cfg.tools.server_health.api_token = ""
    cfg.tools.home_assistant.access_token = ""
    cfg.tools.obsidian.api_key = ""
    cfg.tools.calendar.api_token = ""
    cfg.tools.email.api_token = ""

    assert make_check_server_health_tool(cfg)({})["error"]["code"] == "missing_credentials"
    assert make_home_assistant_read_tool(cfg)({})["error"]["code"] == "missing_credentials"
    assert make_obsidian_search_tool(cfg)({"query": "ops"})["error"]["code"] == "missing_credentials"
    assert make_calendar_summary_tool(cfg)({})["error"]["code"] == "missing_credentials"
    assert make_email_summary_tool(cfg)({})["error"]["code"] == "missing_credentials"


def test_read_only_tools_request_failures(monkeypatch) -> None:
    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        raise RuntimeError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    cfg = _config()

    assert make_check_server_health_tool(cfg)({})["error"]["code"] == "request_failed"
    assert make_home_assistant_read_tool(cfg)({})["error"]["code"] == "request_failed"
    assert make_obsidian_search_tool(cfg)({"query": "ops"})["error"]["code"] == "request_failed"
    assert make_calendar_summary_tool(cfg)({})["error"]["code"] == "request_failed"
    assert make_email_summary_tool(cfg)({})["error"]["code"] == "request_failed"


def test_read_only_tools_success_normalization(monkeypatch) -> None:
    def _fake_urlopen(request: urllib.request.Request, timeout: int = 0):  # noqa: ARG001
        url = request.full_url
        if "health.local" in url:
            return _FakeResponse({"temp_c": 42.2, "load": 33, "disk_used_percent": 71, "status": "ok"})
        if "/api/states" in url:
            return _FakeResponse([
                {
                    "entity_id": "sensor.office_temp",
                    "state": "21.3",
                    "attributes": {"friendly_name": "Office Temp", "unit_of_measurement": "°C"},
                    "last_changed": "2026-01-01T10:00:00+00:00",
                }
            ])
        if "obsidian.local" in url:
            return _FakeResponse({"results": [{"path": "Ops/Runbook.md", "title": "Runbook", "snippet": "Restart sequence"}]})
        if "calendar.local" in url:
            return _FakeResponse({"events": [{"title": "Standup", "start": "09:00", "end": "09:15", "location": "Meet"}]})
        return _FakeResponse({"messages": [{"from": "boss@example.com", "subject": "Status", "received_at": "08:00", "snippet": "Need update"}]})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    cfg = _config()

    health = make_check_server_health_tool(cfg)({})
    ha = make_home_assistant_read_tool(cfg)({})
    obsidian = make_obsidian_search_tool(cfg)({"query": "runbook"})
    calendar = make_calendar_summary_tool(cfg)({})
    email = make_email_summary_tool(cfg)({})

    assert health["ok"] is True
    assert health["data"]["summary"][0]["temperature_c"] == 42.2
    assert health["metadata"]["read_only"] is True
    assert ha["data"]["entities"][0]["friendly_name"] == "Office Temp"
    assert obsidian["data"]["results"][0]["path"] == "Ops/Runbook.md"
    assert calendar["data"]["events"][0]["title"] == "Standup"
    assert email["data"]["unread_count"] == 1
