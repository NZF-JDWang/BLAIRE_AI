from __future__ import annotations

import json
import os
from dataclasses import replace

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

