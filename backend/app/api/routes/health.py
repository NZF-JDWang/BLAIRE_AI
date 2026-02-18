from fastapi import APIRouter

from app.core.config import get_settings
from app.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service="blaire-backend",
        environment=settings.app_env,
    )

