from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import Settings
from app.models.runtime_config import (
    RuntimeConfigAuditEvent,
    RuntimeConfigBundle,
    RuntimeConfigEffective,
    RuntimeConfigOverrides,
    RuntimeConfigUpdateRequest,
)


class RuntimeConfigService:
    def __init__(self, database_url: str):
        self._database_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    def _effective_from_overrides(self, settings: Settings, overrides: RuntimeConfigOverrides) -> RuntimeConfigEffective:
        return RuntimeConfigEffective.from_values(
            search_mode_default=overrides.search_mode_default or settings.search_mode_default,
            sensitive_actions_enabled=(
                overrides.sensitive_actions_enabled
                if overrides.sensitive_actions_enabled is not None
                else settings.sensitive_actions_enabled
            ),
            approval_token_ttl_minutes=overrides.approval_token_ttl_minutes or settings.approval_token_ttl_minutes,
            allowed_network_hosts=overrides.allowed_network_hosts or settings.allowed_network_hosts,
            allowed_network_tools=overrides.allowed_network_tools or settings.allowed_network_tools,
            allowed_obsidian_paths=overrides.allowed_obsidian_paths or settings.allowed_obsidian_paths,
            allowed_ha_operations=overrides.allowed_ha_operations or settings.allowed_ha_operations,
            allowed_homelab_operations=overrides.allowed_homelab_operations or settings.allowed_homelab_operations,
        )

    async def init_schema(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS runtime_config (
            singleton_id BOOLEAN PRIMARY KEY DEFAULT TRUE,
            search_mode_default TEXT,
            sensitive_actions_enabled BOOLEAN,
            approval_token_ttl_minutes INTEGER,
            allowed_network_hosts TEXT,
            allowed_network_tools TEXT,
            allowed_obsidian_paths TEXT,
            allowed_ha_operations TEXT,
            allowed_homelab_operations TEXT,
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS runtime_config_audit (
            id BIGSERIAL PRIMARY KEY,
            actor TEXT NOT NULL,
            previous_overrides JSONB NOT NULL,
            new_overrides JSONB NOT NULL,
            event_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)

    async def schema_ready(self) -> bool:
        query = """
        SELECT
            to_regclass('public.runtime_config') IS NOT NULL AS runtime_config_ready,
            to_regclass('public.runtime_config_audit') IS NOT NULL AS runtime_audit_ready;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                row = await cur.fetchone()
        if row is None:
            return False
        return bool(row["runtime_config_ready"] and row["runtime_audit_ready"])

    async def get_overrides(self) -> RuntimeConfigOverrides:
        query = """
        SELECT
            search_mode_default,
            sensitive_actions_enabled,
            approval_token_ttl_minutes,
            allowed_network_hosts,
            allowed_network_tools,
            allowed_obsidian_paths,
            allowed_ha_operations,
            allowed_homelab_operations,
            updated_by,
            updated_at
        FROM runtime_config
        WHERE singleton_id = TRUE;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                row = await cur.fetchone()
        if row is None:
            return RuntimeConfigOverrides()
        return RuntimeConfigOverrides(**row)

    async def upsert(self, *, actor: str, request: RuntimeConfigUpdateRequest) -> RuntimeConfigOverrides:
        previous = await self.get_overrides()
        query = """
        INSERT INTO runtime_config (
            singleton_id,
            search_mode_default,
            sensitive_actions_enabled,
            approval_token_ttl_minutes,
            allowed_network_hosts,
            allowed_network_tools,
            allowed_obsidian_paths,
            allowed_ha_operations,
            allowed_homelab_operations,
            updated_by,
            updated_at
        )
        VALUES (
            TRUE,
            %(search_mode_default)s,
            %(sensitive_actions_enabled)s,
            %(approval_token_ttl_minutes)s,
            %(allowed_network_hosts)s,
            %(allowed_network_tools)s,
            %(allowed_obsidian_paths)s,
            %(allowed_ha_operations)s,
            %(allowed_homelab_operations)s,
            %(updated_by)s,
            NOW()
        )
        ON CONFLICT (singleton_id) DO UPDATE
        SET search_mode_default = EXCLUDED.search_mode_default,
            sensitive_actions_enabled = EXCLUDED.sensitive_actions_enabled,
            approval_token_ttl_minutes = EXCLUDED.approval_token_ttl_minutes,
            allowed_network_hosts = EXCLUDED.allowed_network_hosts,
            allowed_network_tools = EXCLUDED.allowed_network_tools,
            allowed_obsidian_paths = EXCLUDED.allowed_obsidian_paths,
            allowed_ha_operations = EXCLUDED.allowed_ha_operations,
            allowed_homelab_operations = EXCLUDED.allowed_homelab_operations,
            updated_by = EXCLUDED.updated_by,
            updated_at = NOW()
        RETURNING
            search_mode_default,
            sensitive_actions_enabled,
            approval_token_ttl_minutes,
            allowed_network_hosts,
            allowed_network_tools,
            allowed_obsidian_paths,
            allowed_ha_operations,
            allowed_homelab_operations,
            updated_by,
            updated_at;
        """
        params: dict[str, Any] = request.model_dump()
        params["updated_by"] = actor

        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise ValueError("Failed to persist runtime configuration")
        updated = RuntimeConfigOverrides(**row)

        audit_query = """
        INSERT INTO runtime_config_audit (actor, previous_overrides, new_overrides)
        VALUES (%(actor)s, %(previous_overrides)s, %(new_overrides)s);
        """
        audit_params = {
            "actor": actor,
            "previous_overrides": Jsonb(previous.model_dump(mode="json")),
            "new_overrides": Jsonb(updated.model_dump(mode="json")),
        }
        async with await psycopg.AsyncConnection.connect(self._database_url, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(audit_query, audit_params)
        return updated

    async def get_effective(self, settings: Settings) -> RuntimeConfigEffective:
        try:
            overrides = await self.get_overrides()
        except Exception:  # noqa: BLE001
            overrides = RuntimeConfigOverrides()
        return self._effective_from_overrides(settings, overrides)

    async def get_bundle(self, settings: Settings) -> RuntimeConfigBundle:
        try:
            overrides = await self.get_overrides()
        except Exception:  # noqa: BLE001
            overrides = RuntimeConfigOverrides()
        effective = self._effective_from_overrides(settings, overrides)
        return RuntimeConfigBundle(effective=effective, overrides=overrides)

    async def list_audit(self, limit: int = 100) -> list[RuntimeConfigAuditEvent]:
        query = """
        SELECT id, actor, previous_overrides, new_overrides, event_time
        FROM runtime_config_audit
        ORDER BY id DESC
        LIMIT %(limit)s;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"limit": limit})
                rows = await cur.fetchall()

        events: list[RuntimeConfigAuditEvent] = []
        for row in rows:
            events.append(
                RuntimeConfigAuditEvent(
                    id=row["id"],
                    actor=row["actor"],
                    previous_overrides=RuntimeConfigOverrides(**row["previous_overrides"]),
                    new_overrides=RuntimeConfigOverrides(**row["new_overrides"]),
                    event_time=row["event_time"],
                )
            )
        return events
