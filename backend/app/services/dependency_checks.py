import asyncio

import httpx

from app.core.config import Settings
from app.models.dependencies import DependencyItem, DependencyStatusResponse


async def _check_http(name: str, url: str, timeout: float = 3.0) -> DependencyItem:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code >= 400:
                return DependencyItem(name=name, ok=False, detail=f"http {response.status_code}")
        return DependencyItem(name=name, ok=True, detail="reachable")
    except Exception:  # noqa: BLE001
        return DependencyItem(name=name, ok=False, detail="unreachable")


async def collect_dependency_status(settings: Settings) -> DependencyStatusResponse:
    checks = [
        _check_http("qdrant", settings.qdrant_url.rstrip("/") + "/collections"),
        _check_http("ollama", settings.ollama_base_url.rstrip("/") + "/api/tags"),
        _check_http("mcp_obsidian", settings.mcp_obsidian_url),
        _check_http("mcp_home_assistant", settings.mcp_ha_url),
        _check_http("searxng", settings.searxng_url.rstrip("/") + "/search?q=test&format=json"),
    ]
    results = await asyncio.gather(*checks)

    brave_ok = bool(settings.brave_api_key and settings.brave_api_key.get_secret_value())
    results.append(
        DependencyItem(
            name="brave_api_key",
            ok=brave_ok,
            detail="configured" if brave_ok else "missing",
        )
    )
    return DependencyStatusResponse(dependencies=results)
