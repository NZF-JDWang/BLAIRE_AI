from fastapi import APIRouter, Depends

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.dependencies import DependencyStatusResponse
from app.services.dependency_checks import collect_dependency_status
from app.services.runtime_config_service import RuntimeConfigService

router = APIRouter(tags=["dependencies"])


@router.get("/health/dependencies", response_model=DependencyStatusResponse)
async def dependency_status(
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> DependencyStatusResponse:
    settings = get_settings()
    runtime_config = await RuntimeConfigService(settings.database_url.get_secret_value()).get_effective(settings)
    return await collect_dependency_status(settings, runtime_config)
