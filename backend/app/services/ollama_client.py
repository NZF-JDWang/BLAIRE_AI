from collections.abc import AsyncIterator
import json
from time import monotonic
from typing import Any

import httpx


class OllamaModelCatalog:
    _cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def __init__(self, base_url: str, timeout_seconds: float = 4.0, ttl_seconds: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._ttl = ttl_seconds

    def get_models(self) -> list[dict[str, Any]]:
        now = monotonic()
        cached = self._cache.get(self._base_url)
        if cached and now - cached[0] <= self._ttl:
            return cached[1]

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            rows = data.get("models", [])
            models: list[dict[str, Any]] = [row for row in rows if isinstance(row, dict) and row.get("name")]
            self._cache[self._base_url] = (now, models)
            return models
        except Exception:  # noqa: BLE001
            if cached:
                return cached[1]
            return []

    def get_model_names(self) -> set[str]:
        return {str(model.get("name", "")).strip() for model in self.get_models() if model.get("name")}


def fetch_installed_model_names(base_url: str, timeout_seconds: float = 4.0) -> list[str]:
    catalog = OllamaModelCatalog(base_url=base_url, timeout_seconds=timeout_seconds, ttl_seconds=30.0)
    return sorted(catalog.get_model_names())


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def stream_chat(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content")
                    if content:
                        yield content
                    if data.get("done"):
                        break

    async def embed(self, model: str, text: str) -> list[float]:
        payload = {
            "model": model,
            "prompt": text,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError("Invalid embedding response from Ollama")
        return [float(value) for value in vector]
