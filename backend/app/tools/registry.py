from typing import Any

from app.core.config import get_settings
from app.services.filesystem_sandbox import FilesystemSandbox, FilesystemSandboxError
from app.tools.base import Tool, ToolSpec


class EchoTool:
    spec = ToolSpec(
        name="echo_text",
        action_class="local_safe",
        description="Echoes provided text for testing tool execution flow.",
    )

    async def run(self, arguments: dict[str, Any], target_host: str | None = None) -> dict[str, Any]:
        return {"echo": str(arguments.get("text", ""))}


class NetworkProbeTool:
    spec = ToolSpec(
        name="network_probe",
        action_class="network_sensitive",
        description="Simulated network probe action requiring explicit approval.",
        requires_target_host=True,
    )

    async def run(self, arguments: dict[str, Any], target_host: str | None = None) -> dict[str, Any]:
        check = str(arguments.get("check", "status"))
        return {
            "target_host": target_host,
            "check": check,
            "result": "simulated_ok",
        }


class FilesystemWriteTool:
    spec = ToolSpec(
        name="filesystem_write",
        action_class="local_sensitive",
        description="Writes text content to an allowlisted path.",
        requires_target_host=False,
    )

    async def run(self, arguments: dict[str, Any], target_host: str | None = None) -> dict[str, Any]:  # noqa: ARG002
        _ = target_host
        target_path = str(arguments.get("path", "")).strip()
        content = str(arguments.get("content", ""))
        if not target_path:
            raise ValueError("path argument is required")
        sandbox = FilesystemSandbox(get_settings().allowed_write_paths_list())
        try:
            safe_path = sandbox.validate_target_path(target_path)
        except FilesystemSandboxError as exc:
            raise ValueError(str(exc)) from exc

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return {"path": str(safe_path), "bytes_written": len(content.encode("utf-8"))}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self.register(EchoTool())
        self.register(NetworkProbeTool())
        self.register(FilesystemWriteTool())

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]
