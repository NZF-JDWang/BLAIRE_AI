from pydantic import BaseModel


class DependencyItem(BaseModel):
    name: str
    ok: bool
    detail: str


class DependencyStatusResponse(BaseModel):
    dependencies: list[DependencyItem]

