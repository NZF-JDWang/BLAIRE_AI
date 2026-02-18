from fastapi import APIRouter

from app.core.config import get_settings
from app.models.dependencies import DependencyStatusResponse
from app.services.dependency_checks import collect_dependency_status

router = APIRouter(tags=["dependencies"])


@router.get("/health/dependencies", response_model=DependencyStatusResponse)
async def dependency_status() -> DependencyStatusResponse:
    return await collect_dependency_status(get_settings())

