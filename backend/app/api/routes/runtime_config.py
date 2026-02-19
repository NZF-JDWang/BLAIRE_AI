from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.runtime_config import RuntimeConfigBundle, RuntimeConfigUpdateRequest
from app.services.runtime_config_service import RuntimeConfigService

router = APIRouter(tags=["runtime-config"])


def _service() -> RuntimeConfigService:
    settings = get_settings()
    return RuntimeConfigService(settings.database_url.get_secret_value())


@router.get("/runtime/config", response_model=RuntimeConfigBundle)
async def get_runtime_config(
    _principal: Principal = Depends(require_roles("admin")),
) -> RuntimeConfigBundle:
    settings = get_settings()
    return await _service().get_bundle(settings)


@router.put("/runtime/config", response_model=RuntimeConfigBundle)
async def update_runtime_config(
    request: RuntimeConfigUpdateRequest,
    principal: Principal = Depends(require_roles("admin")),
) -> RuntimeConfigBundle:
    settings = get_settings()
    service = _service()
    try:
        await service.upsert(actor=principal.subject, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return await service.get_bundle(settings)
