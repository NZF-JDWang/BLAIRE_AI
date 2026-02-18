from datetime import datetime

from pydantic import BaseModel, Field

from app.models.chat import ModelClass
from app.models.search import SearchMode


class PreferenceResponse(BaseModel):
    subject: str
    search_mode: SearchMode
    model_class: ModelClass
    model_override: str | None = None
    updated_at: datetime


class PreferenceUpdateRequest(BaseModel):
    search_mode: SearchMode
    model_class: ModelClass
    model_override: str | None = Field(default=None, max_length=128)

