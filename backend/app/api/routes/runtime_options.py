from fastapi import APIRouter

from app.core.config import get_settings
from app.models.runtime_options import RuntimeOptionsResponse
from app.services.model_router import ModelRouter
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["runtime"])


@router.get("/runtime/options", response_model=RuntimeOptionsResponse)
async def runtime_options() -> RuntimeOptionsResponse:
    settings = get_settings()
    router_service = ModelRouter(settings)
    registry = ToolRegistry()
    return RuntimeOptionsResponse(
        search_modes=["brave_only", "searxng_only", "auto_fallback", "parallel"],
        default_search_mode=settings.search_mode_default,
        model_allowlist=router_service.get_allowlist(),
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
