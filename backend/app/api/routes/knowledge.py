from fastapi import APIRouter, Depends

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.knowledge import (
    KnowledgeCitation,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeRetrieveRequest,
    KnowledgeRetrieveResponse,
    KnowledgeStatusResponse,
)
from app.rag.ingestion import DropFolderIngestionService
from app.rag.qdrant_client import QdrantHealthClient
from app.rag.retrieval import IngestionPipeline, RetrievalService
from app.rag.vector_store import QdrantVectorStore
from app.services.ollama_client import OllamaClient

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
    settings = get_settings()
    pipeline = IngestionPipeline(
        ollama_client=OllamaClient(settings.ollama_base_url),
        vector_store=QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection_name),
        embedding_model=settings.model_embedding_default,
    )
    result = await ingestion.ingest_with_pipeline(
        pipeline=pipeline,
        full_rescan=request.full_rescan,
        limit=request.limit,
    )
    return KnowledgeIngestResponse(
        accepted_files=result.accepted_files,
        skipped_files=result.skipped_files,
        started_at=result.started_at,
        chunks_indexed=result.chunks_indexed,
    )


@router.post("/knowledge/retrieve", response_model=KnowledgeRetrieveResponse)
async def retrieve_knowledge(
    request: KnowledgeRetrieveRequest,
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> KnowledgeRetrieveResponse:
    settings = get_settings()
    service = RetrievalService(
        ollama_client=OllamaClient(settings.ollama_base_url),
        vector_store=QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection_name),
        embedding_model=settings.model_embedding_default,
    )
    results = await service.retrieve(query=request.query, limit=request.limit)
    return KnowledgeRetrieveResponse(
        query=request.query,
        citations=[
            KnowledgeCitation(
                source_path=item.source_path,
                source_name=item.source_name,
                file_type=item.file_type,
                chunk_index=item.chunk_index,
                score=item.score,
                text=item.text,
                last_modified=item.last_modified,
            )
            for item in results
        ],
    )
