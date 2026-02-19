import os
import platform
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CallRequest(BaseModel):
    method: str = Field(default="homelab.call")
    params: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="BLAIRE Homelab MCP")


def _allowed_http_hosts() -> set[str]:
    raw = os.getenv("HOMELAB_ALLOWED_HTTP_HOSTS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _ensure_allowed_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http/https URLs are supported")

    host = (parsed.hostname or "").lower()
    allowed = _allowed_http_hosts()
    if allowed and host not in allowed:
        raise HTTPException(status_code=403, detail="HTTP target host is not allowlisted")
    return url


def _system_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "utc_time": datetime.now(timezone.utc).isoformat(),
    }


def _dns_resolve(payload: dict[str, Any]) -> dict[str, Any]:
    host = str(payload.get("host", "")).strip()
    if not host:
        raise HTTPException(status_code=400, detail="payload.host is required")
    try:
        addrinfo = socket.getaddrinfo(host, None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"DNS resolve failed: {exc}") from exc

    addresses: list[str] = []
    for entry in addrinfo:
        ip = entry[4][0]
        if ip not in addresses:
            addresses.append(ip)
    return {"host": host, "addresses": addresses}


def _tcp_check(payload: dict[str, Any]) -> dict[str, Any]:
    host = str(payload.get("host", "")).strip()
    port = int(payload.get("port", 0))
    timeout = float(payload.get("timeout_seconds", 3.0))
    if not host:
        raise HTTPException(status_code=400, detail="payload.host is required")
    if port <= 0 or port > 65535:
        raise HTTPException(status_code=400, detail="payload.port must be 1-65535")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
    except Exception as exc:  # noqa: BLE001
        return {"host": host, "port": port, "ok": False, "detail": str(exc)}
    finally:
        sock.close()
    return {"host": host, "port": port, "ok": True}


async def _http_check(payload: dict[str, Any]) -> dict[str, Any]:
    url = _ensure_allowed_url(str(payload.get("url", "")).strip())
    method = str(payload.get("method", "GET")).upper()
    timeout = float(payload.get("timeout_seconds", 8.0))
    headers = payload.get("headers", {})
    if not isinstance(headers, dict):
        raise HTTPException(status_code=400, detail="payload.headers must be an object")

    request_kwargs: dict[str, Any] = {"headers": headers}
    if "json" in payload:
        request_kwargs["json"] = payload["json"]
    if "body" in payload:
        request_kwargs["content"] = str(payload["body"])

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, **request_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HTTP check failed: {exc}") from exc

    body_preview = response.text[:400]
    return {
        "url": url,
        "method": method,
        "status_code": response.status_code,
        "ok": response.status_code < 400,
        "response_headers": dict(response.headers),
        "body_preview": body_preview,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "homelab-mcp",
        "operations": ["system.info", "dns.resolve", "tcp.check", "http.check"],
        "allowed_http_hosts": sorted(_allowed_http_hosts()),
    }


@app.post("/call")
async def call(request: CallRequest) -> dict[str, Any]:
    if request.method != "homelab.call":
        raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")

    operation = str(request.params.get("operation", "")).strip()
    payload = request.params.get("payload", {})
    if not operation:
        raise HTTPException(status_code=400, detail="params.operation is required")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="params.payload must be an object")

    if operation == "system.info":
        result = _system_info()
    elif operation == "dns.resolve":
        result = _dns_resolve(payload)
    elif operation == "tcp.check":
        result = _tcp_check(payload)
    elif operation == "http.check":
        result = await _http_check(payload)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")

    return {"ok": True, "operation": operation, "result": result}
