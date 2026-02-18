from fastapi import APIRouter, Depends

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.knowledge import KnowledgeIngestRequest, KnowledgeIngestResponse, KnowledgeStatusResponse
from app.rag.ingestion import DropFolderIngestionService
from app.rag.qdrant_client import QdrantHealthClient

router = APIRouter(tags=["knowledge"])

_ingestion_service: DropFolderIngestionService | None = None


def _get_ingestion_service() -> DropFolderIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DropFolderIngestionService(get_settings().drop_folder)
    return _ingestion_service


@router.get("/knowledge/status", response_model=KnowledgeStatusResponse)
async def knowledge_status(
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> KnowledgeStatusResponse:
    settings = get_settings()
    ingestion = _get_ingestion_service()
    files, _ = ingestion.scan_files(limit=1000)
    reachable = await QdrantHealthClient(settings.qdrant_url).is_reachable()
    return KnowledgeStatusResponse(
        drop_folder=settings.drop_folder,
        files_detected=len(files),
        last_scan_at=ingestion.last_scan_at,
        qdrant_reachable=reachable,
    )


@router.post("/knowledge/ingest", response_model=KnowledgeIngestResponse)
async def trigger_ingestion(
    request: KnowledgeIngestRequest,
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> KnowledgeIngestResponse:
    ingestion = _get_ingestion_service()
    result = ingestion.ingest(full_rescan=request.full_rescan, limit=request.limit)
    return KnowledgeIngestResponse(
        accepted_files=result.accepted_files,
        skipped_files=result.skipped_files,
        started_at=result.started_at,
    )
