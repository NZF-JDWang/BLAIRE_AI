from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.models.preferences import PreferenceResponse, PreferenceUpdateRequest


class PreferencesService:
    def __init__(self, database_url: str):
        self._database_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    async def init_schema(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS user_preferences (
            subject TEXT PRIMARY KEY,
            search_mode TEXT NOT NULL,
            model_class TEXT NOT NULL,
            model_override TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)

    async def get_or_default(
        self,
        *,
        subject: str,
        default_search_mode: str,
        default_model_class: str = "general",
    ) -> PreferenceResponse:
        query = """
        SELECT subject, search_mode, model_class, model_override, updated_at
        FROM user_preferences
        WHERE subject = %(subject)s;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"subject": subject})
                row = await cur.fetchone()
        if row:
            return PreferenceResponse(**row)
        return PreferenceResponse(
            subject=subject,
            search_mode=default_search_mode,  # type: ignore[arg-type]
            model_class=default_model_class,  # type: ignore[arg-type]
            model_override=None,
            updated_at=datetime.now(timezone.utc),
        )

    async def upsert(self, *, subject: str, request: PreferenceUpdateRequest) -> PreferenceResponse:
        query = """
        INSERT INTO user_preferences (subject, search_mode, model_class, model_override, updated_at)
        VALUES (%(subject)s, %(search_mode)s, %(model_class)s, %(model_override)s, NOW())
        ON CONFLICT (subject) DO UPDATE
        SET search_mode = EXCLUDED.search_mode,
            model_class = EXCLUDED.model_class,
            model_override = EXCLUDED.model_override,
            updated_at = NOW()
        RETURNING subject, search_mode, model_class, model_override, updated_at;
        """
        params: dict[str, Any] = {
            "subject": subject,
            "search_mode": request.search_mode,
            "model_class": request.model_class,
            "model_override": request.model_override,
        }
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise ValueError("Failed to persist preferences")
        return PreferenceResponse(**row)

