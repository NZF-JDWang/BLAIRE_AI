from pydantic import BaseModel


class RuntimeOptionsResponse(BaseModel):
    search_modes: list[str]
    default_search_mode: str
    model_allowlist: dict[str, list[str]]
    available_models: list[str]
    available_models_by_class: dict[str, list[str]]
    sensitive_actions_enabled: bool
    approval_token_ttl_minutes: int
    allowed_network_hosts: list[str]
    allowed_network_tools: list[str]
    tools: list[dict[str, str | bool]]
