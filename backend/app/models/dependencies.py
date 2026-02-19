from pydantic import BaseModel


class DependencyItem(BaseModel):
    name: str
    ok: bool
    detail: str
    required: bool
    enabled: bool


class DependencyStatusResponse(BaseModel):
    dependencies: list[DependencyItem]
