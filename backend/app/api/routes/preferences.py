from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.preferences import PreferenceResponse, PreferenceUpdateRequest
from app.services.preferences_service import PreferencesService

router = APIRouter(tags=["preferences"])


def _service() -> PreferencesService:
    settings = get_settings()
    return PreferencesService(settings.database_url.get_secret_value())


@router.get("/preferences/me", response_model=PreferenceResponse)
async def get_my_preferences(
    principal: Principal = Depends(require_roles("admin", "user")),
) -> PreferenceResponse:
    settings = get_settings()
    return await _service().get_or_default(
        subject=principal.subject,
        default_search_mode=settings.search_mode_default,
    )


@router.put("/preferences/me", response_model=PreferenceResponse)
async def update_my_preferences(
    request: PreferenceUpdateRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> PreferenceResponse:
    try:
        return await _service().upsert(subject=principal.subject, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

