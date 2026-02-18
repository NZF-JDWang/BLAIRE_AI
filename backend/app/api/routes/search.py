from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.search import SearchRequest, SearchResponse
from app.services.search_service import SearchError, SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    service = SearchService(get_settings())
    try:
        return await service.search(query=request.query, mode=request.mode, limit=request.limit)
    except SearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

