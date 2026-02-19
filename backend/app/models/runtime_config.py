from datetime import datetime

from pydantic import BaseModel, Field, field_validator


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class RuntimeConfigOverrideFields(BaseModel):
    search_mode_default: str | None = Field(default=None)
    sensitive_actions_enabled: bool | None = Field(default=None)
    approval_token_ttl_minutes: int | None = Field(default=None, ge=1, le=120)
    allowed_network_hosts: str | None = Field(default=None)
    allowed_network_tools: str | None = Field(default=None)
    allowed_obsidian_paths: str | None = Field(default=None)
    allowed_ha_operations: str | None = Field(default=None)
    allowed_homelab_operations: str | None = Field(default=None)

    @field_validator("search_mode_default")
    @classmethod
    def validate_search_mode(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = {"brave_only", "searxng_only", "auto_fallback", "parallel"}
        if value not in allowed:
            raise ValueError(f"search_mode_default must be one of {sorted(allowed)}")
        return value


class RuntimeConfigOverrides(RuntimeConfigOverrideFields):
    updated_by: str | None = None
    updated_at: datetime | None = None


class RuntimeConfigUpdateRequest(RuntimeConfigOverrideFields):
    pass


class RuntimeConfigEffective(BaseModel):
    search_mode_default: str
    sensitive_actions_enabled: bool
    approval_token_ttl_minutes: int
    allowed_network_hosts: list[str]
    allowed_network_tools: list[str]
    allowed_obsidian_paths: list[str]
    allowed_ha_operations: list[str]
    allowed_homelab_operations: list[str]

    @classmethod
    def from_values(
        cls,
        *,
        search_mode_default: str,
        sensitive_actions_enabled: bool,
        approval_token_ttl_minutes: int,
        allowed_network_hosts: str,
        allowed_network_tools: str,
        allowed_obsidian_paths: str,
        allowed_ha_operations: str,
        allowed_homelab_operations: str,
    ) -> "RuntimeConfigEffective":
        return cls(
            search_mode_default=search_mode_default,
            sensitive_actions_enabled=sensitive_actions_enabled,
            approval_token_ttl_minutes=approval_token_ttl_minutes,
            allowed_network_hosts=_csv_list(allowed_network_hosts),
            allowed_network_tools=_csv_list(allowed_network_tools),
            allowed_obsidian_paths=_csv_list(allowed_obsidian_paths),
            allowed_ha_operations=_csv_list(allowed_ha_operations),
            allowed_homelab_operations=_csv_list(allowed_homelab_operations),
        )


class RuntimeConfigBundle(BaseModel):
    effective: RuntimeConfigEffective
    overrides: RuntimeConfigOverrides
