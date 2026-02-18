from dataclasses import dataclass
from typing import Any, Literal, Protocol

ActionClass = Literal["local_safe", "local_sensitive", "network_sensitive"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    action_class: ActionClass
    description: str
    requires_target_host: bool = False


class Tool(Protocol):
    spec: ToolSpec

    async def run(self, arguments: dict[str, Any], target_host: str | None = None) -> dict[str, Any]:
        ...

