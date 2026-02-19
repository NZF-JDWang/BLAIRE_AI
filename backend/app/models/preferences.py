from datetime import datetime

from pydantic import BaseModel, Field

from app.models.chat import ModelClass
from app.models.search import SearchMode


class PreferenceResponse(BaseModel):
    subject: str
    search_mode: SearchMode
    model_class: ModelClass
    model_override: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    context_window_tokens: int | None = Field(default=None, ge=256, le=262144)
    use_rag: bool = True
    retrieval_k: int = Field(default=4, ge=1, le=12)
    updated_at: datetime


class PreferenceUpdateRequest(BaseModel):
    search_mode: SearchMode
    model_class: ModelClass
    model_override: str | None = Field(default=None, max_length=128)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    context_window_tokens: int | None = Field(default=None, ge=256, le=262144)
    use_rag: bool = True
    retrieval_k: int = Field(default=4, ge=1, le=12)
