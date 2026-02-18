import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import ORJSONResponse, StreamingResponse

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import ChatRequest, ChatResponse
from app.services.model_router import ModelRouter
from app.services.ollama_client import OllamaClient

router = APIRouter(tags=["chat"])


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    _principal: Principal = Depends(require_roles("admin", "user")),
):
    settings = get_settings()
    logger = get_logger(component="chat")
    model_router = ModelRouter(settings)

    try:
        selection = model_router.select_model(request.model_class, request.model_override)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "chat_request_received",
        model_class=request.model_class,
        model_name=selection.model_name,
        selection_reason=selection.reason,
        message_count=len(request.messages),
    )

    formatted_messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]
    client = OllamaClient(settings.ollama_base_url)

    if request.stream:
        async def event_stream() -> AsyncIterator[str]:
            combined = []
            try:
                yield _sse_event(
                    "meta",
                    {
                        "model": selection.model_name,
                        "model_class": selection.model_class,
                        "selection_reason": selection.reason,
                    },
                )
                async for token in client.stream_chat(selection.model_name, formatted_messages):
                    combined.append(token)
                    yield _sse_event("token", {"text": token})
                yield _sse_event("done", {"text": "".join(combined)})
            except Exception:
                logger.exception("chat_stream_failed")
                yield _sse_event("error", {"message": "Model stream failed"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        chunks = []
        async for token in client.stream_chat(selection.model_name, formatted_messages):
            chunks.append(token)
    except Exception as exc:
        logger.exception("chat_request_failed")
        raise HTTPException(status_code=502, detail="Model request failed") from exc

    return ORJSONResponse(
        ChatResponse(
            model=selection.model_name,
            text="".join(chunks),
        ).model_dump()
    )
