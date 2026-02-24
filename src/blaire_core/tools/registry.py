"""Tool registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


ToolCallable = Callable[[dict], dict]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    risk_level: str
    fn: ToolCallable


class ToolRegistry:
    """Simple in-memory registry."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.risk_level != "safe":
            raise ValueError("v0.1 only allows safe tools")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

