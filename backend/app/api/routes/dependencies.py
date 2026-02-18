from fastapi import APIRouter, Depends

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.dependencies import DependencyStatusResponse
from app.services.dependency_checks import collect_dependency_status

router = APIRouter(tags=["dependencies"])


@router.get("/health/dependencies", response_model=DependencyStatusResponse)
async def dependency_status(
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> DependencyStatusResponse:
    return await collect_dependency_status(get_settings())
