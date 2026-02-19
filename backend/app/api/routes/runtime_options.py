from fastapi import APIRouter, Depends

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.runtime_options import ModelsResponse, RuntimeOptionsResponse
from app.services.model_router import ModelRouter
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["runtime"])


@router.get("/runtime/options", response_model=RuntimeOptionsResponse)
async def runtime_options(
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> RuntimeOptionsResponse:
    settings = get_settings()
    router_service = ModelRouter(settings)
    registry = ToolRegistry()
    model_allowlist = router_service.get_allowlist()
    return RuntimeOptionsResponse(
        search_modes=["brave_only", "searxng_only", "auto_fallback", "parallel"],
        default_search_mode=settings.search_mode_default,
        model_allowlist=model_allowlist,
        available_models=sorted({model for models in model_allowlist.values() for model in models}),
        available_models_by_class=model_allowlist,
        sensitive_actions_enabled=settings.sensitive_actions_enabled,
        approval_token_ttl_minutes=settings.approval_token_ttl_minutes,
        allowed_network_hosts=settings.allowed_network_hosts_list(),
        allowed_network_tools=settings.allowed_network_tools_list(),
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
