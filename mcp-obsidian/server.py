from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VAULT_ROOT = Path("/vault").resolve()


class CallRequest(BaseModel):
    method: str = Field(default="vault.read")
    params: dict = Field(default_factory=dict)


app = FastAPI(title="BLAIRE Obsidian MCP")


def _resolve_relative_path(raw_path: str) -> Path:
    candidate = raw_path.strip().replace("\\", "/")
    if not candidate:
        raise HTTPException(status_code=400, detail="Path cannot be empty")

    normalized = PurePosixPath(candidate)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed")

    target = (VAULT_ROOT / str(normalized)).resolve()
    try:
        target.relative_to(VAULT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Resolved path is outside vault root") from exc
    return target


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "obsidian-mcp",
        "vault_root": str(VAULT_ROOT),
        "vault_exists": VAULT_ROOT.exists(),
    }


@app.post("/call")
def call(request: CallRequest) -> dict:
    if request.method == "vault.read":
        raw_path = str(request.params.get("path", ""))
        target = _resolve_relative_path(raw_path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        content = target.read_text(encoding="utf-8")
        modified_at = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).isoformat()
        return {
            "ok": True,
            "method": request.method,
            "path": raw_path,
            "content": content,
            "bytes": len(content.encode("utf-8")),
            "modified_at": modified_at,
        }

    if request.method == "vault.write":
        raw_path = str(request.params.get("path", ""))
        content = str(request.params.get("content", ""))
        target = _resolve_relative_path(raw_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        modified_at = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).isoformat()
        return {
            "ok": True,
            "method": request.method,
            "path": raw_path,
            "bytes": len(content.encode("utf-8")),
            "modified_at": modified_at,
        }

    raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")
