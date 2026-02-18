from collections.abc import AsyncIterator
import json

import httpx


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
