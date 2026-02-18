from typing import Literal

from pydantic import BaseModel, Field, field_validator


Role = Literal["system", "user", "assistant"]
ModelClass = Literal["general", "vision", "embedding", "code"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    model_class: ModelClass | None = None
    model_override: str | None = Field(default=None, max_length=128)
    stream: bool = True
    use_rag: bool = True
    retrieval_k: int = Field(default=4, ge=1, le=12)

    @field_validator("messages")
    @classmethod
    def require_user_message(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not any(message.role == "user" for message in messages):
            raise ValueError("At least one user message is required")
        return messages


class ChatResponse(BaseModel):
    model: str
    text: str
    citations: list[dict] = Field(default_factory=list)
    rag_status: str = "disabled"
    rag_error: str | None = None
