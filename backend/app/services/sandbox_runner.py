import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter


class SandboxRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class SandboxExecutionRecord:
    command: str
    args: list[str]
    exit_code: int
    stdout: str
    stderr: str
    started_at: str
    duration_ms: float
    timeout_seconds: int

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "args": self.args,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "timeout_seconds": self.timeout_seconds,
        }


class LocalSandboxRunner:
    def __init__(self, allowed_commands: list[str]):
        self._allow = set(allowed_commands)

    async def run(self, command: str, args: list[str], timeout_seconds: int = 10) -> SandboxExecutionRecord:
        if command not in self._allow:
            raise SandboxRunnerError("Command is not allowlisted")

        started = datetime.now(timezone.utc).isoformat()
        started_perf = perf_counter()
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
            exit_code = int(process.returncode or 0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise SandboxRunnerError("Sandbox command timed out")

        duration_ms = round((perf_counter() - started_perf) * 1000, 2)
        return SandboxExecutionRecord(
            command=command,
            args=args,
            exit_code=exit_code,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            started_at=started,
            duration_ms=duration_ms,
            timeout_seconds=timeout_seconds,
        )
