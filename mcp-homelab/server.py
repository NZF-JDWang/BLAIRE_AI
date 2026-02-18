from fastapi import FastAPI
from pydantic import BaseModel, Field


class CallRequest(BaseModel):
    method: str = Field(default="homelab.call")
    params: dict = Field(default_factory=dict)


app = FastAPI(title="BLAIRE Homelab MCP")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "homelab-mcp"}


@app.post("/call")
def call(request: CallRequest) -> dict:
    operation = str(request.params.get("operation", "unknown"))
    payload = request.params.get("payload", {})
    # Stub server for safe, explicit operation contracts.
    return {"ok": True, "operation": operation, "payload": payload, "note": "stub_homelab_mcp"}
