from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.knowledge import (
    KnowledgeCitation,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    ObsidianReindexRequest,
    ObsidianReindexResponse,
    KnowledgeRetrieveRequest,
    KnowledgeRetrieveResponse,
    KnowledgeStatusResponse,
)
from app.rag.ingestion import DropFolderIngestionService
from app.rag.ingestion import SUPPORTED_EXTENSIONS
from app.rag.obsidian_indexer import ObsidianVaultIndexer
from app.rag.qdrant_client import QdrantHealthClient
from app.rag.retrieval import IngestionPipeline, RetrievalService
from app.rag.vector_store import QdrantVectorStore
from app.services.ollama_client import OllamaClient

router = APIRouter(tags=["knowledge"])

_ingestion_service: DropFolderIngestionService | None = None
_obsidian_indexer: ObsidianVaultIndexer | None = None


def _get_ingestion_service() -> DropFolderIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DropFolderIngestionService(get_settings().drop_folder)
    return _ingestion_service


def _get_obsidian_indexer() -> ObsidianVaultIndexer:
    global _obsidian_indexer
    if _obsidian_indexer is None:
        _obsidian_indexer = ObsidianVaultIndexer(get_settings().obsidian_vault_path)
    return _obsidian_indexer


@router.get("/knowledge/status", response_model=KnowledgeStatusResponse)
async def knowledge_status(
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> KnowledgeStatusResponse:
    settings = get_settings()
    ingestion = _get_ingestion_service()
    obsidian = _get_obsidian_indexer()
    files, _ = ingestion.scan_files(limit=1000)
    obsidian_files = obsidian.scan_markdown_files(limit=10000)
    reachable = await QdrantHealthClient(settings.qdrant_url).is_reachable()
    return KnowledgeStatusResponse(
        drop_folder=settings.drop_folder,
        files_detected=len(files),
        last_scan_at=ingestion.last_scan_at,
        qdrant_reachable=reachable,
        obsidian_vault_path=settings.obsidian_vault_path,
        obsidian_files_detected=len(obsidian_files),
        obsidian_last_scan_at=obsidian.last_scan_at,
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
    if request.use_watcher:
        watched = await ingestion.ingest_changed_with_retry(
            pipeline=pipeline,
            limit=request.limit,
            debounce_seconds=request.debounce_seconds,
            retry_base_seconds=request.retry_base_seconds,
            retry_max_seconds=request.retry_max_seconds,
        )
        return KnowledgeIngestResponse(
            accepted_files=watched.scanned_files,
            skipped_files=watched.skipped_files,
            started_at=watched.started_at,
            chunks_indexed=watched.chunks_indexed,
            indexed_files=watched.indexed_files,
            failed_files=watched.failed_files,
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
                ingested_at=item.ingested_at,
            )
            for item in results
        ],
    )


@router.post("/knowledge/upload")
async def upload_to_drop_folder(
    file: UploadFile = File(...),
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> dict[str, str | int]:
    settings = get_settings()
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    drop_dir = Path(settings.drop_folder)
    drop_dir.mkdir(parents=True, exist_ok=True)
    stamped_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}-{filename}"
    target = drop_dir / stamped_name
    target.write_bytes(content)
    return {"stored_filename": stamped_name, "bytes": len(content)}


@router.post("/knowledge/obsidian/reindex", response_model=ObsidianReindexResponse)
async def reindex_obsidian_vault(
    request: ObsidianReindexRequest,
    _principal: Principal = Depends(require_roles("admin", "user")),
) -> ObsidianReindexResponse:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection_name)
    pipeline = IngestionPipeline(
        ollama_client=OllamaClient(settings.ollama_base_url),
        vector_store=vector_store,
        embedding_model=settings.model_embedding_default,
    )
    result = await _get_obsidian_indexer().reindex(
        pipeline=pipeline,
        vector_store=vector_store,
        full_rescan=request.full_rescan,
        limit=request.limit,
    )
    return ObsidianReindexResponse(
        scanned_files=result.scanned_files,
        indexed_files=result.indexed_files,
        unchanged_files=result.unchanged_files,
        skipped_files=result.skipped_files,
        failed_files=result.failed_files,
        chunks_indexed=result.chunks_indexed,
        started_at=result.started_at,
    )
