import subprocess

import pytest

from app.services.cli_sandbox import CliSandboxError, CliSandboxRunner


def test_cli_sandbox_blocks_non_allowlisted_command() -> None:
    runner = CliSandboxRunner(backend="firejail", allowed_commands=["echo"])
    with pytest.raises(CliSandboxError, match="not allowlisted"):
        runner.run(command="python", args=["-V"])


def test_cli_sandbox_firejail_executes(monkeypatch) -> None:
    class FakeProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    captured: dict[str, list[str]] = {}

    def fake_which(name: str):  # noqa: ANN001, ANN202
        mapping = {
            "firejail": "/usr/bin/firejail",
            "echo": "/usr/bin/echo",
        }
        return mapping.get(name)

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
        captured["cmd"] = cmd
        _ = (args, kwargs)
        return FakeProcess()

    monkeypatch.setattr("app.services.cli_sandbox.which", fake_which)
    monkeypatch.setattr("app.services.cli_sandbox.subprocess.run", fake_run)

    runner = CliSandboxRunner(backend="firejail", allowed_commands=["echo"])
    record = runner.run(command="echo", args=["hello"], timeout_seconds=5)
    assert record.backend == "firejail"
    assert record.exit_code == 0
    assert record.stdout == "ok"
    assert captured["cmd"][:5] == ["firejail", "--quiet", "--net=none", "--private", "--"]
    assert captured["cmd"][5] == "/usr/bin/echo"


def test_cli_sandbox_timeout(monkeypatch) -> None:
    def fake_which(name: str):  # noqa: ANN001, ANN202
        mapping = {
            "firejail": "/usr/bin/firejail",
            "echo": "/usr/bin/echo",
        }
        return mapping.get(name)

    monkeypatch.setattr("app.services.cli_sandbox.which", fake_which)

    def _raise_timeout(*args, **kwargs):  # noqa: ANN001, ANN202
        raise subprocess.TimeoutExpired(cmd=["firejail", "echo"], timeout=1)

    monkeypatch.setattr("app.services.cli_sandbox.subprocess.run", _raise_timeout)
    runner = CliSandboxRunner(backend="firejail", allowed_commands=["echo"])
    with pytest.raises(CliSandboxError, match="timed out"):
        runner.run(command="echo", args=["hello"], timeout_seconds=1)


def test_cli_sandbox_bubblewrap_executes(monkeypatch) -> None:
    class FakeProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    captured: dict[str, list[str]] = {}

    def fake_which(name: str):  # noqa: ANN001, ANN202
        mapping = {
            "bwrap": "/usr/bin/bwrap",
            "echo": "/usr/bin/echo",
        }
        return mapping.get(name)

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
        captured["cmd"] = cmd
        _ = (args, kwargs)
        return FakeProcess()

    monkeypatch.setattr("app.services.cli_sandbox.which", fake_which)
    monkeypatch.setattr("app.services.cli_sandbox.subprocess.run", fake_run)

    runner = CliSandboxRunner(backend="bubblewrap", allowed_commands=["echo"])
    record = runner.run(command="echo", args=["hello"], timeout_seconds=5)
    assert record.backend == "bubblewrap"
    assert record.exit_code == 0
    assert "--unshare-net" in captured["cmd"]
    assert "--" in captured["cmd"]
    assert captured["cmd"][-2] == "/usr/bin/echo"
