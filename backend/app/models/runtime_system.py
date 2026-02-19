from pydantic import BaseModel


class RuntimeSystemSummaryResponse(BaseModel):
    app_env: str
    api_docs_enabled: bool
    enable_mcp_services: bool
    enable_vllm: bool
    inference_base_url: str
    qdrant_url: str
    searxng_url: str
    mcp_obsidian_url: str
    mcp_ha_url: str
    mcp_homelab_url: str
    drop_folder: str
    obsidian_vault_path: str
    model_general_default: str
    model_vision_default: str
    model_embedding_default: str
    model_code_default: str | None
    brave_api_key_configured: bool
    telegram_configured: bool
    google_oauth_configured: bool
    imap_configured: bool
    restart_required_note: str
