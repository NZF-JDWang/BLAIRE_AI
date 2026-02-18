from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
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
    ollama_base_url: str = Field(alias="OLLAMA_BASE_URL")

    search_mode_default: str = Field(default="searxng_only", alias="SEARCH_MODE_DEFAULT")
    brave_api_key: SecretStr | None = Field(default=None, alias="BRAVE_API_KEY")
    searxng_url: str = Field(default="http://searxng:8080", alias="SEARXNG_URL")

    mcp_obsidian_url: str = Field(alias="MCP_OBSIDIAN_URL")
    mcp_ha_url: str = Field(alias="MCP_HA_URL")

    model_general_default: str = Field(alias="MODEL_GENERAL_DEFAULT")
    model_vision_default: str = Field(alias="MODEL_VISION_DEFAULT")
    model_embedding_default: str = Field(alias="MODEL_EMBEDDING_DEFAULT")
    model_code_default: str | None = Field(default=None, alias="MODEL_CODE_DEFAULT")

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

    def allowed_hosts_list(self) -> list[str]:
        parsed = [host.strip() for host in self.api_allowed_hosts.split(",") if host.strip()]
        if not parsed:
            raise ValueError("API_ALLOWED_HOSTS cannot be empty")
        return parsed


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
