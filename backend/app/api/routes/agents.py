from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.models.agent import ResearchRequest, ResearchResponse
from app.services.agent_swarm import AgentSwarmService
from app.services.preferences_service import PreferencesService
from app.services.search_service import SearchError, SearchService

router = APIRouter(tags=["agents"])


@router.post("/agents/research", response_model=ResearchResponse)
async def run_research(
    request: ResearchRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
) -> ResearchResponse:
    settings = get_settings()
    prefs = await PreferencesService(settings.database_url.get_secret_value()).get_or_default(
        subject=principal.subject,
        default_search_mode=settings.search_mode_default,
    )
    mode = request.search_mode or prefs.search_mode
    search = SearchService(settings)
    swarm = AgentSwarmService(
        search,
        max_tool_calls=settings.agent_max_tool_calls,
        max_recursion_depth=settings.agent_max_recursion_depth,
        worker_timeout_seconds=settings.agent_worker_timeout_seconds,
        overall_timeout_seconds=settings.agent_overall_timeout_seconds,
    )
    try:
        return await swarm.run_research(query=request.query, search_mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
