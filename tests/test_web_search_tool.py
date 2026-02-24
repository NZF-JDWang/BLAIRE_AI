from __future__ import annotations

import json
import os
import urllib.request

from blaire_core.config import (
    AppConfig,
    AppSection,
    HeartbeatSection,
    LLMSection,
    LoggingSection,
    PathsSection,
    PromptSection,
    SessionMaintenanceSection,
    SessionSection,
    ToolsSection,
    WebSearchSection,
)
import blaire_core.tools.builtin_tools as builtin_tools
from blaire_core.tools.builtin_tools import make_web_search_tool


def _config(api_key: str = "") -> AppConfig:
    return AppConfig(
        app=AppSection(env="dev"),
        paths=PathsSection(data_root="./data", log_dir="data/logs"),
        llm=LLMSection(base_url="http://localhost:11434", model="m", timeout_seconds=30),
        heartbeat=HeartbeatSection(interval_seconds=0),
        tools=ToolsSection(
            web_search=WebSearchSection(
                api_key=api_key,
                timeout_seconds=10,
                cache_ttl_minutes=15,
                result_count=10,
                safesearch="off",
            )
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


def test_web_search_missing_key() -> None:
    if "BLAIRE_BRAVE_API_KEY" in os.environ:
        del os.environ["BLAIRE_BRAVE_API_KEY"]
    tool = make_web_search_tool(_config(api_key=""))
    result = tool({"query": "test"})
    assert result["ok"] is False
    assert result["error"]["code"] == "missing_brave_api_key"


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.status = 200
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)


def test_web_search_wraps_external_content_and_caches(monkeypatch) -> None:
    call_count = {"n": 0}

    def _fake_urlopen(request: urllib.request.Request, timeout: int = 0):  # noqa: ARG001
        _ = request
        call_count["n"] += 1
        return _FakeResponse(
            {
                "web": {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "description": "A snippet from the web.",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    builtin_tools._WEB_CACHE.clear()

    tool = make_web_search_tool(_config(api_key="test-key"))
    first = tool({"query": "hello world", "count": 1})
    second = tool({"query": "hello world", "count": 1})

    assert first["ok"] is True
    assert first["metadata"]["cached"] is False
    assert second["ok"] is True
    assert second["metadata"]["cached"] is True
    assert call_count["n"] == 1
    snippet = first["data"]["results"][0]["snippet"]
    ext = first["data"]["results"][0]["external_content"]
    assert "BLAIRE_UNTRUSTED_CONTENT" in snippet
    assert ext["untrusted"] is True
    assert ext["source"] == "web_search"
    assert ext["wrapped"] is True
