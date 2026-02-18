from dataclasses import dataclass
from datetime import datetime, timezone
from shutil import which
import subprocess


class CliSandboxError(ValueError):
    pass


@dataclass(frozen=True)
class CliSandboxRecord:
    command: str
    args: list[str]
    backend: str
    exit_code: int
    stdout: str
    stderr: str
    started_at: str
    timeout_seconds: int

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "args": self.args,
            "backend": self.backend,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "started_at": self.started_at,
            "timeout_seconds": self.timeout_seconds,
        }


class CliSandboxRunner:
    def __init__(self, *, backend: str, allowed_commands: list[str]):
        self._backend = backend
        self._allow = set(allowed_commands)

    def _resolve_command(self, command: str) -> str:
        resolved = which(command)
        if not resolved:
            raise CliSandboxError(f"Command '{command}' is not available on PATH")
        return resolved

    def _sandbox_prefix(self) -> list[str]:
        if self._backend == "firejail":
            if which("firejail") is None:
                raise CliSandboxError("firejail is not installed")
            return ["firejail", "--quiet", "--net=none", "--private", "--"]
        if self._backend == "bubblewrap":
            bwrap = which("bwrap") or which("bubblewrap")
            if bwrap is None:
                raise CliSandboxError("bubblewrap is not installed")
            return [
                bwrap,
                "--die-with-parent",
                "--unshare-net",
                "--unshare-pid",
                "--new-session",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/bin",
                "/bin",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--chdir",
                "/tmp",
                "--",
            ]
        raise CliSandboxError("Unsupported sandbox backend")

    def run(self, *, command: str, args: list[str], timeout_seconds: int = 10) -> CliSandboxRecord:
        if command not in self._allow:
            raise CliSandboxError("Command is not allowlisted")

        started_at = datetime.now(timezone.utc).isoformat()
        resolved_command = self._resolve_command(command)
        sandbox_cmd = [*self._sandbox_prefix(), resolved_command, *args]
        try:
            proc = subprocess.run(  # noqa: S603
                sandbox_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CliSandboxError("CLI sandbox command timed out") from exc
        except Exception as exc:  # noqa: BLE001
            raise CliSandboxError(f"CLI sandbox execution failed: {exc}") from exc

        return CliSandboxRecord(
            command=command,
            args=args,
            backend=self._backend,
            exit_code=int(proc.returncode),
            stdout=proc.stdout,
            stderr=proc.stderr,
            started_at=started_at,
            timeout_seconds=timeout_seconds,
        )
