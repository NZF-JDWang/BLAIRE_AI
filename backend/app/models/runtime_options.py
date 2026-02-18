from pydantic import BaseModel


class RuntimeOptionsResponse(BaseModel):
    search_modes: list[str]
    default_search_mode: str
    model_allowlist: dict[str, list[str]]

