from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, Header, HTTPException

from app.core.config import Settings, get_settings

Role = Literal["admin", "user"]


@dataclass(frozen=True)
class Principal:
    role: Role
    subject: str


def get_principal(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if not settings.require_auth:
        return Principal(role="admin", subject="dev-bypass")

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if x_api_key in settings.admin_api_keys_list():
        return Principal(role="admin", subject="admin")
    if x_api_key in settings.user_api_keys_list():
        return Principal(role="user", subject="user")
    raise HTTPException(status_code=403, detail="Invalid API key")


def require_roles(*roles: Role) -> Callable[[Principal], Principal]:
    def _guard(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return principal

    return _guard

