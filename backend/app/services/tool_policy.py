from app.core.config import Settings
from app.tools.base import ToolSpec


class ToolPolicyError(ValueError):
    pass


class ToolPolicy:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._allowed_hosts = set(settings.allowed_network_hosts_list())
        self._allowed_tools = set(settings.allowed_network_tools_list())

    def validate_network_tool(self, spec: ToolSpec, target_host: str | None) -> None:
        if spec.action_class != "network_sensitive":
            return
        if not self._settings.sensitive_actions_enabled:
            raise ToolPolicyError("Sensitive actions are globally disabled")
        if not target_host:
            raise ToolPolicyError("target_host is required for network-sensitive tools")
        if self._allowed_hosts and target_host not in self._allowed_hosts:
            raise ToolPolicyError("Target host is not allowlisted")
        if self._allowed_tools and spec.name not in self._allowed_tools:
            raise ToolPolicyError("Tool is not allowlisted for network execution")

