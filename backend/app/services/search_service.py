import asyncio
from typing import Any

import httpx

from app.core.config import Settings
from app.models.search import SearchMode, SearchResponse, SearchResult


class SearchError(RuntimeError):
    pass


class SearchService:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def search(self, query: str, mode: SearchMode | None = None, limit: int = 10) -> SearchResponse:
        selected_mode: SearchMode = mode or self._settings.search_mode_default  # type: ignore[assignment]

        if selected_mode == "searxng_only":
            results = await self._search_searxng(query, limit)
            return SearchResponse(mode=selected_mode, results=results, providers_used=["searxng"])

        if selected_mode == "brave_only":
            results = await self._search_brave(query, limit)
            return SearchResponse(mode=selected_mode, results=results, providers_used=["brave"])

        if selected_mode == "auto_fallback":
            try:
                results = await self._search_searxng(query, limit)
                return SearchResponse(mode=selected_mode, results=results, providers_used=["searxng"])
            except SearchError:
                results = await self._search_brave(query, limit)
                return SearchResponse(mode=selected_mode, results=results, providers_used=["brave"])

        searx_task = self._search_searxng(query, limit)
        brave_task = self._search_brave(query, limit)
        searx_results, brave_results = await asyncio.gather(searx_task, brave_task, return_exceptions=True)

        merged: list[SearchResult] = []
        providers_used: list[str] = []

        if not isinstance(searx_results, Exception):
            providers_used.append("searxng")
            merged.extend(searx_results)
        if not isinstance(brave_results, Exception):
            providers_used.append("brave")
            merged.extend(brave_results)

        if not merged:
            raise SearchError("All search providers failed")

        return SearchResponse(
            mode=selected_mode,
            results=self._dedupe_results(merged, limit),
            providers_used=providers_used,
        )

    async def _search_searxng(self, query: str, limit: int) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "language": "en-US",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self._settings.searxng_url.rstrip("/") + "/search", params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise SearchError("SearxNG request failed") from exc

        results = payload.get("results", [])[:limit]
        normalized: list[SearchResult] = []
        for item in results:
            normalized.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                    provider="searxng",
                )
            )
        if not normalized:
            raise SearchError("SearxNG returned no results")
        return normalized

    async def _search_brave(self, query: str, limit: int) -> list[SearchResult]:
        key = self._settings.brave_api_key.get_secret_value() if self._settings.brave_api_key else ""
        if not key:
            raise SearchError("Brave API key not configured")

        headers = {
            "X-Subscription-Token": key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": limit,
            "search_lang": "en",
            "country": "us",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise SearchError("Brave request failed") from exc

        raw_results = payload.get("web", {}).get("results", [])[:limit]
        normalized: list[SearchResult] = []
        for item in raw_results:
            normalized.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("description", "")),
                    provider="brave",
                )
            )
        if not normalized:
            raise SearchError("Brave returned no results")
        return normalized

    def _dedupe_results(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for result in results:
            if result.url in seen:
                continue
            seen.add(result.url)
            deduped.append(result)
            if len(deduped) >= limit:
                break
        return deduped

