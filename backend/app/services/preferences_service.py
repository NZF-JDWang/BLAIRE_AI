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
            temperature DOUBLE PRECISION NOT NULL DEFAULT 0.7,
            top_p DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            max_tokens INTEGER,
            context_window_tokens INTEGER,
            use_rag BOOLEAN NOT NULL DEFAULT TRUE,
            retrieval_k INTEGER NOT NULL DEFAULT 4,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS temperature DOUBLE PRECISION NOT NULL DEFAULT 0.7;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS top_p DOUBLE PRECISION NOT NULL DEFAULT 1.0;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS max_tokens INTEGER;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS context_window_tokens INTEGER;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS use_rag BOOLEAN NOT NULL DEFAULT TRUE;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS retrieval_k INTEGER NOT NULL DEFAULT 4;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)

    async def schema_ready(self) -> bool:
        query = "SELECT to_regclass('public.user_preferences') IS NOT NULL AS ready;"
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                row = await cur.fetchone()
        if row is None:
            return False
        return bool(row["ready"])

    async def get_or_default(
        self,
        *,
        subject: str,
        default_search_mode: str,
        default_model_class: str = "general",
    ) -> PreferenceResponse:
        query = """
        SELECT subject, search_mode, model_class, model_override, temperature, top_p, max_tokens, context_window_tokens, use_rag, retrieval_k, updated_at
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
            temperature=0.7,
            top_p=1.0,
            max_tokens=None,
            context_window_tokens=None,
            use_rag=True,
            retrieval_k=4,
            updated_at=datetime.now(timezone.utc),
        )

    async def upsert(self, *, subject: str, request: PreferenceUpdateRequest) -> PreferenceResponse:
        query = """
        INSERT INTO user_preferences (
            subject,
            search_mode,
            model_class,
            model_override,
            temperature,
            top_p,
            max_tokens,
            context_window_tokens,
            use_rag,
            retrieval_k,
            updated_at
        )
        VALUES (
            %(subject)s,
            %(search_mode)s,
            %(model_class)s,
            %(model_override)s,
            %(temperature)s,
            %(top_p)s,
            %(max_tokens)s,
            %(context_window_tokens)s,
            %(use_rag)s,
            %(retrieval_k)s,
            NOW()
        )
        ON CONFLICT (subject) DO UPDATE
        SET search_mode = EXCLUDED.search_mode,
            model_class = EXCLUDED.model_class,
            model_override = EXCLUDED.model_override,
            temperature = EXCLUDED.temperature,
            top_p = EXCLUDED.top_p,
            max_tokens = EXCLUDED.max_tokens,
            context_window_tokens = EXCLUDED.context_window_tokens,
            use_rag = EXCLUDED.use_rag,
            retrieval_k = EXCLUDED.retrieval_k,
            updated_at = NOW()
        RETURNING subject, search_mode, model_class, model_override, temperature, top_p, max_tokens, context_window_tokens, use_rag, retrieval_k, updated_at;
        """
        params: dict[str, Any] = {
            "subject": subject,
            "search_mode": request.search_mode,
            "model_class": request.model_class,
            "model_override": request.model_override,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "context_window_tokens": request.context_window_tokens,
            "use_rag": request.use_rag,
            "retrieval_k": request.retrieval_k,
        }
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise ValueError("Failed to persist preferences")
        return PreferenceResponse(**row)
