import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from app.models.approval import ActionClass, ApprovalAuditEvent, ApprovalRecord


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _to_record(row: dict[str, Any]) -> ApprovalRecord:
    return ApprovalRecord(**row)


class ApprovalService:
    def __init__(self, database_url: str):
        self._database_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    async def init_schema(self) -> None:
        create_approvals = """
        CREATE TABLE IF NOT EXISTS approvals (
            id UUID PRIMARY KEY,
            status TEXT NOT NULL,
            action_class TEXT NOT NULL,
            target_host TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            action_payload JSONB NOT NULL,
            payload_hash TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            approved_by TEXT,
            approval_token_hash TEXT,
            token_expires_at TIMESTAMPTZ,
            executed_at TIMESTAMPTZ,
            rejection_reason TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT approvals_status_check CHECK (status IN ('pending','approved','rejected','executed','expired')),
            CONSTRAINT approvals_action_class_check CHECK (action_class IN ('local_safe','local_sensitive','network_sensitive'))
        );
        """
        create_audit = """
        CREATE TABLE IF NOT EXISTS approval_audit_events (
            id BIGSERIAL PRIMARY KEY,
            approval_id UUID,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL,
            details JSONB NOT NULL DEFAULT '{}'::jsonb,
            event_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(create_approvals)
                await cur.execute(create_audit)

    async def create_pending(
        self,
        *,
        approval_id: UUID,
        action_class: ActionClass,
        target_host: str,
        tool_name: str,
        action_payload: dict[str, Any],
        requested_by: str,
    ) -> ApprovalRecord:
        payload_hash = canonical_payload_hash(action_payload)
        query = """
        INSERT INTO approvals (
            id, status, action_class, target_host, tool_name, action_payload, payload_hash, requested_by
        ) VALUES (
            %(id)s, 'pending', %(action_class)s, %(target_host)s, %(tool_name)s, %(action_payload)s::jsonb, %(payload_hash)s, %(requested_by)s
        )
        RETURNING *;
        """
        params = {
            "id": approval_id,
            "action_class": action_class,
            "target_host": target_host,
            "tool_name": tool_name,
            "action_payload": json.dumps(action_payload),
            "payload_hash": payload_hash,
            "requested_by": requested_by,
        }
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                row = await cur.fetchone()
                await self._insert_audit(
                    conn,
                    approval_id=approval_id,
                    event_type="created",
                    actor=requested_by,
                    details={"tool_name": tool_name, "target_host": target_host, "action_class": action_class},
                )
            await conn.commit()
        if row is None:
            raise ValueError("Failed to create approval")
        return _to_record(row)

    async def get_approval(self, approval_id: UUID) -> ApprovalRecord | None:
        query = """
        SELECT id, status, action_class, target_host, tool_name, action_payload, payload_hash, requested_by,
               approved_by, created_at, updated_at, token_expires_at, executed_at, rejection_reason
        FROM approvals
        WHERE id = %(id)s;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"id": approval_id})
                row = await cur.fetchone()
        return _to_record(row) if row else None

    async def approve(self, approval_id: UUID, actor: str, ttl_minutes: int = 10) -> tuple[ApprovalRecord, str, datetime]:
        token = secrets.token_urlsafe(32)
        token_hash = hash_token(token)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
        query = """
        UPDATE approvals
        SET status = 'approved',
            approved_by = %(actor)s,
            approval_token_hash = %(token_hash)s,
            token_expires_at = %(expires_at)s,
            rejection_reason = NULL,
            updated_at = NOW()
        WHERE id = %(id)s
          AND status = 'pending'
        RETURNING id, status, action_class, target_host, tool_name, action_payload, payload_hash, requested_by,
                  approved_by, created_at, updated_at, token_expires_at, executed_at, rejection_reason;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    {
                        "id": approval_id,
                        "actor": actor,
                        "token_hash": token_hash,
                        "expires_at": expires_at,
                    },
                )
                row = await cur.fetchone()
                if row is None:
                    raise ValueError("Approval is not in pending state")
                await self._insert_audit(
                    conn,
                    approval_id=approval_id,
                    event_type="approved",
                    actor=actor,
                    details={"token_expires_at": expires_at.isoformat()},
                )
            await conn.commit()
        return _to_record(row), token, expires_at

    async def reject(self, approval_id: UUID, actor: str, reason: str) -> ApprovalRecord:
        query = """
        UPDATE approvals
        SET status = 'rejected',
            approved_by = %(actor)s,
            rejection_reason = %(reason)s,
            approval_token_hash = NULL,
            token_expires_at = NULL,
            updated_at = NOW()
        WHERE id = %(id)s
          AND status IN ('pending', 'approved')
        RETURNING id, status, action_class, target_host, tool_name, action_payload, payload_hash, requested_by,
                  approved_by, created_at, updated_at, token_expires_at, executed_at, rejection_reason;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"id": approval_id, "actor": actor, "reason": reason})
                row = await cur.fetchone()
                if row is None:
                    raise ValueError("Approval cannot be rejected in its current state")
                await self._insert_audit(
                    conn,
                    approval_id=approval_id,
                    event_type="rejected",
                    actor=actor,
                    details={"reason": reason},
                )
            await conn.commit()
        return _to_record(row)

    async def execute(
        self,
        approval_id: UUID,
        actor: str,
        execution_token: str,
        expected_payload_hash: str,
    ) -> ApprovalRecord:
        query = """
        SELECT id, status, payload_hash, approval_token_hash, token_expires_at
        FROM approvals
        WHERE id = %(id)s;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"id": approval_id})
                state = await cur.fetchone()
                if state is None:
                    raise ValueError("Approval not found")
                if state["status"] != "approved":
                    raise ValueError("Approval is not in approved state")
                expires = state["token_expires_at"]
                if expires is None or expires <= datetime.now(timezone.utc):
                    await self._expire(conn, approval_id, actor)
                    raise ValueError("Approval token expired")
                if state["payload_hash"] != expected_payload_hash:
                    raise ValueError("Payload hash mismatch")
                if state["approval_token_hash"] != hash_token(execution_token):
                    raise ValueError("Execution token mismatch")

                update_query = """
                UPDATE approvals
                SET status = 'executed',
                    approval_token_hash = NULL,
                    token_expires_at = NULL,
                    executed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %(id)s
                  AND status = 'approved'
                RETURNING id, status, action_class, target_host, tool_name, action_payload, payload_hash, requested_by,
                          approved_by, created_at, updated_at, token_expires_at, executed_at, rejection_reason;
                """
                await cur.execute(update_query, {"id": approval_id})
                row = await cur.fetchone()
                if row is None:
                    raise ValueError("Approval execution failed")
                await self._insert_audit(
                    conn,
                    approval_id=approval_id,
                    event_type="executed",
                    actor=actor,
                    details={"payload_hash": expected_payload_hash},
                )
            await conn.commit()
        return _to_record(row)

    async def list_pending(self, limit: int = 50) -> list[ApprovalRecord]:
        query = """
        SELECT id, status, action_class, target_host, tool_name, action_payload, payload_hash, requested_by,
               approved_by, created_at, updated_at, token_expires_at, executed_at, rejection_reason
        FROM approvals
        WHERE status IN ('pending', 'approved')
        ORDER BY created_at DESC
        LIMIT %(limit)s;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"limit": limit})
                rows = await cur.fetchall()
        return [_to_record(row) for row in rows]

    async def list_recent(self, limit: int = 100) -> list[ApprovalRecord]:
        query = """
        SELECT id, status, action_class, target_host, tool_name, action_payload, payload_hash, requested_by,
               approved_by, created_at, updated_at, token_expires_at, executed_at, rejection_reason
        FROM approvals
        ORDER BY created_at DESC
        LIMIT %(limit)s;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"limit": limit})
                rows = await cur.fetchall()
        return [_to_record(row) for row in rows]

    async def list_audit_events(self, approval_id: UUID, limit: int = 200) -> list[ApprovalAuditEvent]:
        query = """
        SELECT id, approval_id, event_type, actor, details, event_time
        FROM approval_audit_events
        WHERE approval_id = %(approval_id)s
        ORDER BY event_time DESC
        LIMIT %(limit)s;
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, {"approval_id": approval_id, "limit": limit})
                rows = await cur.fetchall()
        return [ApprovalAuditEvent(**row) for row in rows]

    async def _expire(self, conn: psycopg.AsyncConnection, approval_id: UUID, actor: str) -> None:
        query = """
        UPDATE approvals
        SET status = 'expired',
            approval_token_hash = NULL,
            updated_at = NOW()
        WHERE id = %(id)s
          AND status = 'approved';
        """
        async with conn.cursor() as cur:
            await cur.execute(query, {"id": approval_id})
        await self._insert_audit(
            conn,
            approval_id=approval_id,
            event_type="expired",
            actor=actor,
            details={},
        )

    async def _insert_audit(
        self,
        conn: psycopg.AsyncConnection,
        *,
        approval_id: UUID,
        event_type: str,
        actor: str,
        details: dict[str, Any],
    ) -> None:
        query = """
        INSERT INTO approval_audit_events (approval_id, event_type, actor, details)
        VALUES (%(approval_id)s, %(event_type)s, %(actor)s, %(details)s::jsonb);
        """
        async with conn.cursor() as cur:
            await cur.execute(
                query,
                {
                    "approval_id": approval_id,
                    "event_type": event_type,
                    "actor": actor,
                    "details": json.dumps(details),
                },
            )
