import asyncio

import httpx

from app.core.config import Settings
from app.models.dependencies import DependencyItem, DependencyStatusResponse


async def _check_http(
    name: str,
    url: str,
    *,
    required: bool,
    enabled: bool,
    timeout: float = 3.0,
) -> DependencyItem:
    if not enabled:
        return DependencyItem(
            name=name,
            ok=True,
            detail="disabled",
            required=required,
            enabled=False,
        )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code >= 400:
                return DependencyItem(
                    name=name,
                    ok=False,
                    detail=f"http {response.status_code}",
                    required=required,
                    enabled=True,
                )
        return DependencyItem(
            name=name,
            ok=True,
            detail="reachable",
            required=required,
            enabled=True,
        )
    except Exception:  # noqa: BLE001
        return DependencyItem(
            name=name,
            ok=False,
            detail="unreachable",
            required=required,
            enabled=True,
        )


async def collect_dependency_status(settings: Settings) -> DependencyStatusResponse:
    search_mode = settings.search_mode_default
    searx_required = search_mode in {"searxng_only", "parallel"}
    brave_required = search_mode in {"brave_only", "parallel"}
    searx_enabled = bool(settings.searxng_url.strip()) and search_mode in {"searxng_only", "auto_fallback", "parallel"}
    brave_configured = bool(settings.brave_api_key and settings.brave_api_key.get_secret_value())
    brave_enabled = brave_configured or brave_required
    inference_enabled = bool(settings.inference_base_url.strip())

    checks = [
        _check_http(
            "qdrant",
            settings.qdrant_url.rstrip("/") + "/collections",
            required=True,
            enabled=True,
        ),
        _check_http(
            "inference_api",
            settings.inference_base_url.rstrip("/") + "/v1/models",
            required=True,
            enabled=inference_enabled,
        ),
        _check_http(
            "vllm",
            settings.vllm_base_url.rstrip("/") + "/health",
            required=False,
            enabled=settings.enable_vllm,
        ),
        _check_http(
            "mcp_obsidian",
            settings.mcp_obsidian_url,
            required=False,
            enabled=settings.enable_mcp_services,
        ),
        _check_http(
            "mcp_home_assistant",
            settings.mcp_ha_url,
            required=False,
            enabled=settings.enable_mcp_services,
        ),
        _check_http(
            "mcp_homelab",
            settings.mcp_homelab_url,
            required=False,
            enabled=settings.enable_mcp_services,
        ),
        _check_http(
            "searxng",
            settings.searxng_url.rstrip("/") + "/search?q=test&format=json",
            required=searx_required,
            enabled=searx_enabled,
        ),
    ]
    results = await asyncio.gather(*checks)

    brave_ok = not brave_enabled or brave_configured
    results.append(
        DependencyItem(
            name="brave_api_key",
            ok=brave_ok,
            detail="configured" if brave_configured else ("missing" if brave_required else "disabled"),
            required=brave_required,
            enabled=brave_enabled,
        )
    )
    return DependencyStatusResponse(dependencies=results)
