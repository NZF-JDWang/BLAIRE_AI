from __future__ import annotations

from blaire_core.config import SSHHostSection, read_config_snapshot
from blaire_core.orchestrator import approve_tool_call, build_context, call_tool
import blaire_core.tools.builtin_tools as builtin_tools


def test_docker_restart_requires_approval_then_executes(monkeypatch, tmp_path) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path), "llm.model": "test-model"})
    assert snapshot.effective_config is not None
    cfg = snapshot.effective_config
    cfg.tools.ssh.command_allowlist = ["docker restart"]
    cfg.tools.ssh.hosts = {
        "bsl1": SSHHostSection(host="127.0.0.1", user="x", port=22, key_path="")
    }
    context = build_context(cfg, snapshot)

    called = {"count": 0}

    def _fake_ssh_exec(host, command: str, timeout_seconds: int):
        _ = host, timeout_seconds
        called["count"] += 1
        assert command.startswith("docker restart")
        return (0, "container restarted\n", "")

    monkeypatch.setattr(builtin_tools, "_ssh_exec", _fake_ssh_exec)

    blocked = call_tool(context, "docker_container_restart", {"host": "bsl1", "container": "jellyfin"})
    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "approval_required"
    token = blocked["metadata"]["approval_token"]

    approved = approve_tool_call(context, token=token, tool_name="docker_container_restart", args={"host": "bsl1", "container": "jellyfin"})
    assert approved["ok"] is True
    assert called["count"] == 1


def test_media_pipeline_status_aggregates_three_services(monkeypatch, tmp_path) -> None:
    snapshot = read_config_snapshot("dev", {"paths.data_root": str(tmp_path), "llm.model": "test-model"})
    assert snapshot.effective_config is not None
    cfg = snapshot.effective_config
    cfg.tools.integrations.sonarr.base_url = "http://sonarr.local"
    cfg.tools.integrations.radarr.base_url = "http://radarr.local"
    cfg.tools.integrations.qbittorrent.base_url = "http://qb.local"

    context = build_context(cfg, snapshot)

    def _fake_http_json(method: str, url: str, timeout_seconds: int, headers=None, body=None):
        _ = method, timeout_seconds, headers, body
        if "sonarr" in url:
            return {"records": [{"title": "Episode"}]}
        if "radarr" in url:
            return {"records": [{"title": "Movie"}]}
        if "qb.local" in url:
            return {"server_state": {"dl_info_speed": 1}}
        return {}

    monkeypatch.setattr(builtin_tools, "_http_json", _fake_http_json)
    monkeypatch.setattr(
        builtin_tools.urllib.request,
        "urlopen",
        lambda *args, **kwargs: type("R", (), {"headers": {}, "__enter__": lambda s: s, "__exit__": lambda s, *x: None})(),
    )

    result = call_tool(context, "media_pipeline_status", {})
    assert result["ok"] is True
    assert "sonarr" in result["data"]
    assert "radarr" in result["data"]
    assert "qbittorrent" in result["data"]
