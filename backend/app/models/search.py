from typing import Literal

from pydantic import BaseModel, Field

SearchMode = Literal["brave_only", "searxng_only", "auto_fallback", "parallel"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: SearchMode | None = None
    limit: int = Field(default=10, ge=1, le=25)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    provider: str


class SearchResponse(BaseModel):
    mode: SearchMode
    results: list[SearchResult]
    providers_used: list[str]

