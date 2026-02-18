import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models.agent import ResearchResponse, SwarmTraceStep, WorkerResult
from app.models.preferences import PreferenceResponse


def _set_required_env() -> None:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["API_ALLOWED_HOSTS"] = "testserver,localhost,127.0.0.1,backend"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["REQUIRE_AUTH"] = "true"
    os.environ["USER_API_KEYS"] = "test-user-key"
    os.environ["ADMIN_API_KEYS"] = "test-admin-key"


_set_required_env()

from app.main import create_app  # noqa: E402
from app.services.agent_swarm import AgentSwarmService  # noqa: E402
from app.services.preferences_service import PreferencesService  # noqa: E402
from app.services.swarm_trace_store import swarm_trace_store  # noqa: E402


def _fake_pref() -> PreferenceResponse:
    return PreferenceResponse(
        subject="user",
        search_mode="searxng_only",
        model_class="general",
        model_override=None,
        updated_at=datetime.now(timezone.utc),
    )


def test_agents_research_records_live_trace(monkeypatch) -> None:
    async def fake_get_or_default(self, *, subject: str, default_search_mode: str, default_model_class: str = "general"):  # noqa: ANN001, ANN202
        _ = (self, subject, default_search_mode, default_model_class)
        return _fake_pref()

    async def fake_run_research(self, query: str, search_mode: str | None = None, recursion_depth: int = 0):  # noqa: ANN001, ANN202
        _ = (self, query, search_mode, recursion_depth)
        return ResearchResponse(
            query="test topic",
            supervisor_summary="summary",
            workers=[WorkerResult(worker_id="worker-1", summary="s", sources=["https://a"])],
            trace=[
                SwarmTraceStep(
                    step="swarm_start",
                    status="started",
                    timestamp=datetime.now(timezone.utc),
                    details={"x": "y"},
                )
            ],
        )

    swarm_trace_store.clear()
    monkeypatch.setattr(PreferencesService, "get_or_default", fake_get_or_default)
    monkeypatch.setattr(AgentSwarmService, "run_research", fake_run_research)

    client = TestClient(create_app())
    response = client.post(
        "/agents/research",
        headers={"X-API-Key": "test-user-key"},
        json={"query": "test topic"},
    )
    assert response.status_code == 200

    live = client.get("/agents/swarm/live", headers={"X-API-Key": "test-user-key"})
    assert live.status_code == 200
    payload = live.json()
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["query"] == "test topic"
    assert payload["runs"][0]["trace"][0]["step"] == "swarm_start"
