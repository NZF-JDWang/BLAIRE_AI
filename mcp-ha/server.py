import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CallRequest(BaseModel):
    method: str = Field(default="ha.call_service")
    params: dict[str, Any] = Field(default_factory=dict)


def _ha_config() -> tuple[str, str, bool]:
    base = os.getenv("HOME_ASSISTANT_URL", "").strip().rstrip("/")
    token = os.getenv("HOME_ASSISTANT_TOKEN", "").strip()
    verify_tls = os.getenv("HOME_ASSISTANT_VERIFY_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
    return base, token, verify_tls


def _resolve_service(operation: str, payload: dict[str, Any]) -> tuple[str, str]:
    if "." in operation:
        domain, service = operation.split(".", 1)
        return domain.strip(), service.strip()
    domain = str(payload.get("domain", "")).strip()
    service = str(payload.get("service", "")).strip()
    return domain, service


app = FastAPI(title="BLAIRE Home Assistant MCP")


@app.get("/health")
async def health() -> dict:
    base, token, verify_tls = _ha_config()
    if not base or not token:
        return {
            "status": "degraded",
            "service": "ha-mcp",
            "detail": "HOME_ASSISTANT_URL/TOKEN not configured",
        }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=8.0, verify=verify_tls) as client:
            response = await client.get(f"{base}/api/", headers=headers)
            if response.status_code >= 400:
                return {
                    "status": "degraded",
                    "service": "ha-mcp",
                    "detail": f"Home Assistant API unavailable ({response.status_code})",
                }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "degraded",
            "service": "ha-mcp",
            "detail": f"Home Assistant health check failed: {exc}",
        }

    return {"status": "ok", "service": "ha-mcp"}


@app.post("/call")
async def call(request: CallRequest) -> dict:
    if request.method != "ha.call_service":
        raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")

    base, token, verify_tls = _ha_config()
    if not base or not token:
        raise HTTPException(status_code=503, detail="Home Assistant MCP is not configured")

    operation = str(request.params.get("operation", "")).strip()
    payload = request.params.get("payload", {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be an object")

    domain, service = _resolve_service(operation, payload)
    if not domain or not service:
        raise HTTPException(status_code=400, detail="Operation must resolve to domain.service")

    request_body = dict(payload)
    request_body.pop("domain", None)
    request_body.pop("service", None)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"{base}/api/services/{domain}/{service}"
    try:
        async with httpx.AsyncClient(timeout=20.0, verify=verify_tls) as client:
            response = await client.post(url, headers=headers, json=request_body)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Home Assistant request failed: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Home Assistant request failed: {exc}") from exc

    return {
        "ok": True,
        "operation": f"{domain}.{service}",
        "payload": request_body,
        "ha_response": data,
    }
