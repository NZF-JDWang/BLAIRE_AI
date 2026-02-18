from fastapi import APIRouter

from app.core.config import get_settings
from app.models.runtime_options import RuntimeOptionsResponse
from app.services.model_router import ModelRouter

router = APIRouter(tags=["runtime"])


@router.get("/runtime/options", response_model=RuntimeOptionsResponse)
async def runtime_options() -> RuntimeOptionsResponse:
    settings = get_settings()
    router_service = ModelRouter(settings)
    return RuntimeOptionsResponse(
        search_modes=["brave_only", "searxng_only", "auto_fallback", "parallel"],
        default_search_mode=settings.search_mode_default,
        model_allowlist=router_service.get_allowlist(),
    )

