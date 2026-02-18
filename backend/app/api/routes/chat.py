import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import ORJSONResponse, StreamingResponse

from app.core.auth import Principal, require_roles
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import ChatRequest, ChatResponse
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import QdrantVectorStore
from app.services.model_router import ModelRouter
from app.services.ollama_client import OllamaClient
from app.services.preferences_service import PreferencesService

router = APIRouter(tags=["chat"])


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    principal: Principal = Depends(require_roles("admin", "user")),
):
    settings = get_settings()
    logger = get_logger(component="chat")
    model_router = ModelRouter(settings)
    prefs = await PreferencesService(settings.database_url.get_secret_value()).get_or_default(
        subject=principal.subject,
        default_search_mode=settings.search_mode_default,
    )
    effective_model_class = request.model_class or prefs.model_class
    effective_override = request.model_override or prefs.model_override

    try:
        selection = model_router.select_model(effective_model_class, effective_override)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "chat_request_received",
        model_class=effective_model_class,
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
    citations: list[dict] = []
    rag_status = "disabled"
    rag_error: str | None = None
    if request.use_rag:
        rag_status = "none"
        try:
            latest_user = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
            if latest_user:
                retrieval = RetrievalService(
                    ollama_client=client,
                    vector_store=QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection_name),
                    embedding_model=settings.model_embedding_default,
                )
                results = await retrieval.retrieve(query=latest_user, limit=request.retrieval_k)
                citations = [
                    {
                        "source_path": item.source_path,
                        "source_name": item.source_name,
                        "file_type": item.file_type,
                        "chunk_index": item.chunk_index,
                        "score": item.score,
                        "text": item.text,
                    }
                    for item in results
                ]
                if citations:
                    rag_status = "used"
                    context_text = "\n\n".join(
                        [f"[{c['source_name']}#{c['chunk_index']}] {c['text']}" for c in citations]
                    )
                    formatted_messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": (
                                "Use the following retrieved knowledge when relevant. "
                                "If unsure, say you are unsure.\n\n"
                                f"{context_text}"
                            ),
                        },
                    )
        except Exception:
            logger.exception("rag_retrieval_failed")
            rag_status = "failed"
            rag_error = "retrieval_unavailable"

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
                        "rag_status": rag_status,
                        "rag_error": rag_error,
                        "citations": citations,
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
            citations=citations,
            rag_status=rag_status,
            rag_error=rag_error,
        ).model_dump()
    )
