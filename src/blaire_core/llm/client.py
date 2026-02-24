"""Ollama client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from blaire_core.config import AppConfig


class OllamaClient:
    """Synchronous Ollama /api/chat client."""

    def __init__(self, config: AppConfig) -> None:
        self._base_url = config.llm.base_url.rstrip("/")
        self._model = config.llm.model
        self._timeout = config.llm.timeout_seconds
        self._temperature = config.llm.temperature
        self._top_p = config.llm.top_p
        self._repeat_penalty = config.llm.repeat_penalty
        self._num_ctx = config.llm.num_ctx

    @property
    def base_url(self) -> str:
        return self._base_url

    def check_reachable(self, timeout_seconds: int = 3) -> tuple[bool, str]:
        url = f"{self._base_url}/api/tags"
        request = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                if 200 <= response.status < 300:
                    return True, "ok"
                return False, f"http_{response.status}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def generate(self, system_prompt: str, messages: list[dict], max_tokens: int) -> str:
        """Generate response text; returns graceful fallback on failures."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": self._temperature,
                "top_p": self._top_p,
                "repeat_penalty": self._repeat_penalty,
                "num_ctx": self._num_ctx,
            },
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                body = json.loads(response.read().decode("utf-8"))
            message = body.get("message", {})
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            return "I could not generate a response right now."
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return "I'm having trouble reaching the local model right now. Please try again."

