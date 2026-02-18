from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.agent import ResearchRequest, ResearchResponse
from app.services.agent_swarm import AgentSwarmService
from app.services.search_service import SearchError, SearchService

router = APIRouter(tags=["agents"])


@router.post("/agents/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    search = SearchService(get_settings())
    swarm = AgentSwarmService(search)
    try:
        return await swarm.run_research(query=request.query, search_mode=request.search_mode)
    except SearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

