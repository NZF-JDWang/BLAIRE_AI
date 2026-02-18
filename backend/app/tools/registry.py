from typing import Any

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


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self.register(EchoTool())
        self.register(NetworkProbeTool())

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

