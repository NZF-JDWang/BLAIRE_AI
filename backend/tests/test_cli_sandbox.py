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

    monkeypatch.setattr("app.services.cli_sandbox.which", lambda name: "/usr/bin/firejail" if name == "firejail" else None)
    monkeypatch.setattr("app.services.cli_sandbox.subprocess.run", lambda *args, **kwargs: FakeProcess())

    runner = CliSandboxRunner(backend="firejail", allowed_commands=["echo"])
    record = runner.run(command="echo", args=["hello"], timeout_seconds=5)
    assert record.backend == "firejail"
    assert record.exit_code == 0
    assert record.stdout == "ok"


def test_cli_sandbox_timeout(monkeypatch) -> None:
    monkeypatch.setattr("app.services.cli_sandbox.which", lambda name: "/usr/bin/firejail" if name == "firejail" else None)

    def _raise_timeout(*args, **kwargs):  # noqa: ANN001, ANN202
        raise subprocess.TimeoutExpired(cmd=["firejail", "echo"], timeout=1)

    monkeypatch.setattr("app.services.cli_sandbox.subprocess.run", _raise_timeout)
    runner = CliSandboxRunner(backend="firejail", allowed_commands=["echo"])
    with pytest.raises(CliSandboxError, match="timed out"):
        runner.run(command="echo", args=["hello"], timeout_seconds=1)
