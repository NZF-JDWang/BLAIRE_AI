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


class ModelsResponse(BaseModel):
    installed_models: list[str]
    allowlist: dict[str, list[str]]
    defaults: dict[str, str | None]
    model_allow_any_inference: bool


class ModelPullRequest(BaseModel):
    model_name: str


class ModelPullResponse(BaseModel):
    status: str
    model_name: str
    detail: str
