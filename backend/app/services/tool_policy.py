from app.core.config import Settings
from app.models.runtime_config import RuntimeConfigEffective
from app.tools.base import ToolSpec


class ToolPolicyError(ValueError):
    pass


class ToolPolicy:
    def __init__(self, settings: Settings, runtime_config: RuntimeConfigEffective | None = None):
        self._settings = settings
        self._sensitive_actions_enabled = (
            runtime_config.sensitive_actions_enabled if runtime_config else settings.sensitive_actions_enabled
        )
        self._allowed_hosts = set(runtime_config.allowed_network_hosts if runtime_config else settings.allowed_network_hosts_list())
        self._allowed_tools = set(runtime_config.allowed_network_tools if runtime_config else settings.allowed_network_tools_list())

    def validate_network_tool(self, spec: ToolSpec, target_host: str | None) -> None:
        if spec.action_class != "network_sensitive":
            return
        if not self._sensitive_actions_enabled:
            raise ToolPolicyError("Sensitive actions are globally disabled")
        if not target_host:
            raise ToolPolicyError("target_host is required for network-sensitive tools")
        if self._allowed_hosts and target_host not in self._allowed_hosts:
            raise ToolPolicyError("Target host is not allowlisted")
        if self._allowed_tools and spec.name not in self._allowed_tools:
            raise ToolPolicyError("Tool is not allowlisted for network execution")
