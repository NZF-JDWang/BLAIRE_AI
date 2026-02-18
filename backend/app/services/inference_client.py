from collections.abc import AsyncIterator
import json
from time import monotonic
from typing import Any

import httpx


class InferenceModelCatalog:
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
                response = client.get(f"{self._base_url}/v1/models")
                response.raise_for_status()
                data = response.json()
            rows = data.get("data", [])
            models: list[dict[str, Any]] = [row for row in rows if isinstance(row, dict) and row.get("id")]
            self._cache[self._base_url] = (now, models)
            return models
        except Exception:  # noqa: BLE001
            if cached:
                return cached[1]
            return []

    def get_model_names(self) -> set[str]:
        return {str(model.get("id", "")).strip() for model in self.get_models() if model.get("id")}


def fetch_available_model_names(base_url: str, timeout_seconds: float = 4.0) -> list[str]:
    catalog = InferenceModelCatalog(base_url=base_url, timeout_seconds=timeout_seconds, ttl_seconds=30.0)
    return sorted(catalog.get_model_names())


class InferenceClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def stream_chat(self, model: str, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_chunk = line[5:].strip()
                    if not data_chunk or data_chunk == "[DONE]":
                        break
                    data = json.loads(data_chunk)
                    delta = ((data.get("choices") or [{}])[0]).get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield content

    async def embed(self, model: str, text: str) -> list[float]:
        payload = {
            "model": model,
            "input": text,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/v1/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        rows = data.get("data") or []
        if not rows or not isinstance(rows[0], dict):
            raise ValueError("Invalid embedding response from inference API")
        vector = rows[0].get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError("Invalid embedding response from inference API")
        return [float(value) for value in vector]
