from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    app_env: str = Field(default="production", alias="APP_ENV")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_docs_enabled: bool = Field(default=False, alias="API_DOCS_ENABLED")
    api_allowed_hosts: str = Field(
        default="localhost,127.0.0.1,backend",
        alias="API_ALLOWED_HOSTS",
    )

    database_url: SecretStr = Field(alias="DATABASE_URL")
    qdrant_url: str = Field(alias="QDRANT_URL")
    inference_base_url: str = Field(
        default="http://localai:8080",
        alias="INFERENCE_BASE_URL",
    )
    localai_models_path: str = Field(default="/models", alias="LOCALAI_MODELS_PATH")
    vllm_base_url: str = Field(default="http://vllm:8000", alias="VLLM_BASE_URL")

    search_mode_default: str = Field(default="searxng_only", alias="SEARCH_MODE_DEFAULT")
    brave_api_key: SecretStr | None = Field(default=None, alias="BRAVE_API_KEY")
    searxng_url: str = Field(default="http://searxng:8080", alias="SEARXNG_URL")

    mcp_obsidian_url: str = Field(alias="MCP_OBSIDIAN_URL")
    mcp_ha_url: str = Field(alias="MCP_HA_URL")
    mcp_homelab_url: str = Field(default="http://homelab-mcp:3000", alias="MCP_HOMELAB_URL")
    enable_mcp_services: bool = Field(default=False, alias="ENABLE_MCP_SERVICES")
    enable_vllm: bool = Field(default=False, alias="ENABLE_VLLM")
    drop_folder: str = Field(default="/app/knowledge/drop", alias="DROP_FOLDER")
    obsidian_vault_path: str = Field(default="/vault", alias="OBSIDIAN_VAULT_PATH")
    sensitive_actions_enabled: bool = Field(default=True, alias="SENSITIVE_ACTIONS_ENABLED")
    approval_token_ttl_minutes: int = Field(default=10, alias="APPROVAL_TOKEN_TTL_MINUTES")
    allowed_network_hosts: str = Field(default="", alias="ALLOWED_NETWORK_HOSTS")
    allowed_network_tools: str = Field(default="", alias="ALLOWED_NETWORK_TOOLS")
    allowed_write_paths: str = Field(default="/app/knowledge/drop", alias="ALLOWED_WRITE_PATHS")
    allowed_obsidian_paths: str = Field(default="", alias="ALLOWED_OBSIDIAN_PATHS")
    allowed_ha_operations: str = Field(default="", alias="ALLOWED_HA_OPERATIONS")
    allowed_homelab_operations: str = Field(default="", alias="ALLOWED_HOMELAB_OPERATIONS")
    require_auth: bool = Field(default=True, alias="REQUIRE_AUTH")
    admin_api_keys: str = Field(default="", alias="ADMIN_API_KEYS")
    user_api_keys: str = Field(default="", alias="USER_API_KEYS")

    model_general_default: str = Field(alias="MODEL_GENERAL_DEFAULT")
    model_vision_default: str = Field(alias="MODEL_VISION_DEFAULT")
    model_embedding_default: str = Field(alias="MODEL_EMBEDDING_DEFAULT")
    model_code_default: str | None = Field(default=None, alias="MODEL_CODE_DEFAULT")
    model_allow_any_inference: bool = Field(
        default=False,
        alias="MODEL_ALLOW_ANY_INFERENCE",
    )
    model_allowlist_extra_general: str = Field(default="", alias="MODEL_ALLOWLIST_EXTRA_GENERAL")
    model_allowlist_extra_vision: str = Field(default="", alias="MODEL_ALLOWLIST_EXTRA_VISION")
    model_allowlist_extra_embedding: str = Field(default="", alias="MODEL_ALLOWLIST_EXTRA_EMBEDDING")
    model_allowlist_extra_code: str = Field(default="", alias="MODEL_ALLOWLIST_EXTRA_CODE")
    model_disallowlist: str = Field(default="", alias="MODEL_DISALLOWLIST")
    qdrant_collection_name: str = Field(default="knowledge_multimodal", alias="QDRANT_COLLECTION_NAME")
    qdrant_embedding_dim: int = Field(default=768, alias="QDRANT_EMBEDDING_DIM")
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB")
    backup_path: str = Field(default="/backups", alias="BACKUP_PATH")
    agent_max_tool_calls: int = Field(default=4, alias="AGENT_MAX_TOOL_CALLS")
    agent_max_recursion_depth: int = Field(default=2, alias="AGENT_MAX_RECURSION_DEPTH")
    agent_worker_timeout_seconds: int = Field(default=12, alias="AGENT_WORKER_TIMEOUT_SECONDS")
    agent_overall_timeout_seconds: int = Field(default=20, alias="AGENT_OVERALL_TIMEOUT_SECONDS")
    sandbox_allowed_commands: str = Field(default="echo", alias="SANDBOX_ALLOWED_COMMANDS")
    cli_sandbox_backend: str = Field(default="firejail", alias="CLI_SANDBOX_BACKEND")
    cli_sandbox_enabled: bool = Field(default=False, alias="CLI_SANDBOX_ENABLED")
    piper_bin: str = Field(default="piper", alias="PIPER_BIN")
    piper_voice_model: str = Field(default="", alias="PIPER_VOICE_MODEL")
    faster_whisper_bin: str = Field(default="faster-whisper", alias="FASTER_WHISPER_BIN")
    faster_whisper_model: str = Field(default="small", alias="FASTER_WHISPER_MODEL")
    telegram_bot_token: SecretStr | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_default_chat_id: str = Field(default="", alias="TELEGRAM_DEFAULT_CHAT_ID")
    telegram_webhook_secret_token: SecretStr | None = Field(default=None, alias="TELEGRAM_WEBHOOK_SECRET_TOKEN")
    google_api_base: str = Field(default="https://www.googleapis.com", alias="GOOGLE_API_BASE")
    google_oauth_token: SecretStr | None = Field(default=None, alias="GOOGLE_OAUTH_TOKEN")
    imap_host: str = Field(default="", alias="IMAP_HOST")
    imap_user: str = Field(default="", alias="IMAP_USER")
    imap_password: SecretStr | None = Field(default=None, alias="IMAP_PASSWORD")

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, value: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if value not in allowed:
            raise ValueError(f"APP_ENV must be one of {sorted(allowed)}")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        normalized = value.lower()
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return normalized

    @field_validator("search_mode_default")
    @classmethod
    def validate_search_mode(cls, value: str) -> str:
        allowed = {"brave_only", "searxng_only", "auto_fallback", "parallel"}
        if value not in allowed:
            raise ValueError(f"SEARCH_MODE_DEFAULT must be one of {sorted(allowed)}")
        return value

    @field_validator("approval_token_ttl_minutes")
    @classmethod
    def validate_approval_ttl(cls, value: int) -> int:
        if value < 1 or value > 120:
            raise ValueError("APPROVAL_TOKEN_TTL_MINUTES must be between 1 and 120")
        return value

    @field_validator("agent_max_tool_calls")
    @classmethod
    def validate_agent_max_tool_calls(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError("AGENT_MAX_TOOL_CALLS must be between 1 and 20")
        return value

    @field_validator("agent_max_recursion_depth")
    @classmethod
    def validate_agent_max_recursion_depth(cls, value: int) -> int:
        if value < 0 or value > 10:
            raise ValueError("AGENT_MAX_RECURSION_DEPTH must be between 0 and 10")
        return value

    @field_validator("agent_worker_timeout_seconds", "agent_overall_timeout_seconds")
    @classmethod
    def validate_agent_timeouts(cls, value: int) -> int:
        if value < 1 or value > 120:
            raise ValueError("Agent timeout values must be between 1 and 120 seconds")
        return value

    @field_validator("cli_sandbox_backend")
    @classmethod
    def validate_cli_sandbox_backend(cls, value: str) -> str:
        allowed = {"firejail", "bubblewrap"}
        if value not in allowed:
            raise ValueError(f"CLI_SANDBOX_BACKEND must be one of {sorted(allowed)}")
        return value

    def allowed_hosts_list(self) -> list[str]:
        parsed = [host.strip() for host in self.api_allowed_hosts.split(",") if host.strip()]
        if not parsed:
            raise ValueError("API_ALLOWED_HOSTS cannot be empty")
        return parsed

    def allowed_network_hosts_list(self) -> list[str]:
        return [host.strip() for host in self.allowed_network_hosts.split(",") if host.strip()]

    def allowed_network_tools_list(self) -> list[str]:
        return [tool.strip() for tool in self.allowed_network_tools.split(",") if tool.strip()]

    def allowed_write_paths_list(self) -> list[str]:
        return [path.strip() for path in self.allowed_write_paths.split(",") if path.strip()]

    def allowed_obsidian_paths_list(self) -> list[str]:
        return [path.strip() for path in self.allowed_obsidian_paths.split(",") if path.strip()]

    def allowed_ha_operations_list(self) -> list[str]:
        return [op.strip() for op in self.allowed_ha_operations.split(",") if op.strip()]

    def allowed_homelab_operations_list(self) -> list[str]:
        return [op.strip() for op in self.allowed_homelab_operations.split(",") if op.strip()]

    def admin_api_keys_list(self) -> list[str]:
        return [key.strip() for key in self.admin_api_keys.split(",") if key.strip()]

    def user_api_keys_list(self) -> list[str]:
        return [key.strip() for key in self.user_api_keys.split(",") if key.strip()]

    def sandbox_allowed_commands_list(self) -> list[str]:
        return [command.strip() for command in self.sandbox_allowed_commands.split(",") if command.strip()]

    def model_allowlist_extra_general_list(self) -> list[str]:
        return [model.strip() for model in self.model_allowlist_extra_general.split(",") if model.strip()]

    def model_allowlist_extra_vision_list(self) -> list[str]:
        return [model.strip() for model in self.model_allowlist_extra_vision.split(",") if model.strip()]

    def model_allowlist_extra_embedding_list(self) -> list[str]:
        return [model.strip() for model in self.model_allowlist_extra_embedding.split(",") if model.strip()]

    def model_allowlist_extra_code_list(self) -> list[str]:
        return [model.strip() for model in self.model_allowlist_extra_code.split(",") if model.strip()]

    def model_disallowlist_list(self) -> list[str]:
        return [model.strip() for model in self.model_disallowlist.split(",") if model.strip()]

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
