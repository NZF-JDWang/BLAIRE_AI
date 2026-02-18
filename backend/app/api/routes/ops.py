from fastapi import APIRouter, Depends

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.ops import BackupRequest, BackupResponse
from app.services.backup_service import BackupService

router = APIRouter(tags=["ops"])


@router.post("/ops/backup", response_model=BackupResponse)
async def run_backup(
    request: BackupRequest,
    _principal: Principal = Depends(require_roles("admin")),
) -> BackupResponse:
    settings = get_settings()
    service = BackupService(
        backup_root=settings.backup_path,
        database_url=settings.database_url.get_secret_value(),
        qdrant_url=settings.qdrant_url,
    )
    return service.run_backup(
        include_postgres=request.include_postgres,
        include_qdrant=request.include_qdrant,
    )

