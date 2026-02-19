from pydantic import BaseModel


class RuntimeDiagnosticsResponse(BaseModel):
    role: str
    require_auth: bool
    enable_mcp_services: bool
    mcp_obsidian_configured: bool
    mcp_ha_configured: bool
    mcp_homelab_configured: bool
    drop_folder_path: str
    drop_folder_exists: bool
    obsidian_vault_path: str
    obsidian_vault_exists: bool
    effective_search_mode_default: str
    effective_sensitive_actions_enabled: bool
    effective_approval_token_ttl_minutes: int
