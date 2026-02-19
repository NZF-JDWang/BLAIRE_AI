from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.search import SearchRequest, SearchResponse
from app.services.preferences_service import PreferencesService
from app.services.runtime_config_service import RuntimeConfigService
from app.services.search_service import SearchError, SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> SearchResponse:
    settings = get_settings()
    runtime_config = await RuntimeConfigService(settings.database_url.get_secret_value()).get_effective(settings)
    prefs = await PreferencesService(settings.database_url.get_secret_value()).get_or_default(
        subject=principal.subject,
        default_search_mode=runtime_config.search_mode_default,
    )
    service = SearchService(settings)
    try:
        return await service.search(query=request.query, mode=request.mode or prefs.search_mode, limit=request.limit)
    except SearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
