import httpx
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.runtime_diagnostics import RuntimeDiagnosticsResponse
from app.models.runtime_options import ModelPullRequest, ModelPullResponse, ModelsResponse, RuntimeOptionsResponse
from app.models.runtime_system import RuntimeSystemSummaryResponse
from app.services.model_router import ModelRouter
from app.services.runtime_config_service import RuntimeConfigService
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["runtime"])


async def _pull_model_via_inference(inference_base_url: str, model_name: str) -> str:
    endpoint = inference_base_url.rstrip("/") + "/api/pull"
    payload = {"model": model_name, "stream": False}
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(endpoint, json=payload)
    if response.status_code >= 400:
        raise ValueError(f"Pull failed with status {response.status_code}")
    return "Pull requested successfully"


@router.get("/runtime/options", response_model=RuntimeOptionsResponse)
async def runtime_options(
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> RuntimeOptionsResponse:
    settings = get_settings()
    runtime_config = await RuntimeConfigService(settings.database_url.get_secret_value()).get_effective(settings)
    router_service = ModelRouter(settings)
    registry = ToolRegistry()
    model_allowlist = router_service.get_allowlist()
    return RuntimeOptionsResponse(
        search_modes=["brave_only", "searxng_only", "auto_fallback", "parallel"],
        default_search_mode=runtime_config.search_mode_default,
        model_allowlist=model_allowlist,
        available_models=sorted({model for models in model_allowlist.values() for model in models}),
        available_models_by_class=model_allowlist,
        sensitive_actions_enabled=runtime_config.sensitive_actions_enabled,
        approval_token_ttl_minutes=runtime_config.approval_token_ttl_minutes,
        allowed_network_hosts=runtime_config.allowed_network_hosts,
        allowed_network_tools=runtime_config.allowed_network_tools,
        tools=[
            {
                "name": spec.name,
                "action_class": spec.action_class,
                "description": spec.description,
                "requires_target_host": spec.requires_target_host,
            }
            for spec in registry.list_specs()
        ],
    )


@router.get("/models", response_model=ModelsResponse)
async def models(
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> ModelsResponse:
    settings = get_settings()
    router_service = ModelRouter(settings)
    allowlist = router_service.get_allowlist()
    return ModelsResponse(
        installed_models=router_service.get_installed_models(),
        allowlist=allowlist,
        defaults={
            "general": settings.model_general_default,
            "vision": settings.model_vision_default,
            "embedding": settings.model_embedding_default,
            "code": settings.model_code_default,
        },
        model_allow_any_inference=settings.model_allow_any_inference,
    )


@router.post("/models/pull", response_model=ModelPullResponse)
async def pull_model(
    request: ModelPullRequest,
    _principal: Principal = Depends(require_roles("admin")),
) -> ModelPullResponse:
    settings = get_settings()
    model_name = request.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    try:
        detail = await _pull_model_via_inference(settings.inference_base_url, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Model pull failed: {exc}") from exc
    return ModelPullResponse(status="accepted", model_name=model_name, detail=detail)


@router.get("/runtime/diagnostics", response_model=RuntimeDiagnosticsResponse)
async def runtime_diagnostics(
    principal: Principal = Depends(require_roles("admin", "user")),
) -> RuntimeDiagnosticsResponse:
    settings = get_settings()
    runtime_config = await RuntimeConfigService(settings.database_url.get_secret_value()).get_effective(settings)
    return RuntimeDiagnosticsResponse(
        role=principal.role,
        require_auth=settings.require_auth,
        enable_mcp_services=settings.enable_mcp_services,
        mcp_obsidian_configured=bool(settings.mcp_obsidian_url.strip()),
        mcp_ha_configured=bool(settings.mcp_ha_url.strip()),
        mcp_homelab_configured=bool(settings.mcp_homelab_url.strip()),
        drop_folder_path=settings.drop_folder,
        drop_folder_exists=Path(settings.drop_folder).exists(),
        obsidian_vault_path=settings.obsidian_vault_path,
        obsidian_vault_exists=Path(settings.obsidian_vault_path).exists(),
        effective_search_mode_default=runtime_config.search_mode_default,
        effective_sensitive_actions_enabled=runtime_config.sensitive_actions_enabled,
        effective_approval_token_ttl_minutes=runtime_config.approval_token_ttl_minutes,
    )


@router.get("/runtime/system-summary", response_model=RuntimeSystemSummaryResponse)
async def runtime_system_summary(
    _principal: Principal = Depends(require_roles("admin")),
) -> RuntimeSystemSummaryResponse:
    settings = get_settings()
    return RuntimeSystemSummaryResponse(
        app_env=settings.app_env,
        api_docs_enabled=settings.api_docs_enabled,
        enable_mcp_services=settings.enable_mcp_services,
        enable_vllm=settings.enable_vllm,
        inference_base_url=settings.inference_base_url,
        qdrant_url=settings.qdrant_url,
        searxng_url=settings.searxng_url,
        mcp_obsidian_url=settings.mcp_obsidian_url,
        mcp_ha_url=settings.mcp_ha_url,
        mcp_homelab_url=settings.mcp_homelab_url,
        drop_folder=settings.drop_folder,
        obsidian_vault_path=settings.obsidian_vault_path,
        model_general_default=settings.model_general_default,
        model_vision_default=settings.model_vision_default,
        model_embedding_default=settings.model_embedding_default,
        model_code_default=settings.model_code_default,
        brave_api_key_configured=bool(settings.brave_api_key and settings.brave_api_key.get_secret_value()),
        telegram_configured=bool(settings.telegram_bot_token and settings.telegram_bot_token.get_secret_value()),
        google_oauth_configured=bool(settings.google_oauth_token and settings.google_oauth_token.get_secret_value()),
        imap_configured=bool(settings.imap_host and settings.imap_user and settings.imap_password),
        restart_required_note="These values come from environment/system config and require service restart when changed.",
    )
