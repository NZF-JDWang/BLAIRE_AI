"""File-backed memory helpers (JSON + SQLite structured store)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import tempfile
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from blaire_core.embeddings import embed_text

logger = logging.getLogger(__name__)


_MEMORY_FILES: dict[str, Any] = {
    "projects.json": [],
    "todo.json": [],
    "debates.json": [],
    "predictions.json": [],
    "patterns.json": [],
}

_MEMORY_TYPES = {"fact", "decision", "preference", "incident", "pattern_candidate"}
_STABILITY_VALUES = {"stable", "evolving", "temp"}


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_tags(tags: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        items = [part.strip() for part in tags.split(",")]
    else:
        items = [str(part).strip() for part in tags]
    out: list[str] = []
    seen: set[str] = set()
    for tag in items:
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(tag)
    return out


def _embedding_to_blob(vector: list[float]) -> bytes:
    if not vector:
        return b""
    return struct.pack(f"<{len(vector)}f", *vector)


def _blob_to_embedding(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    if len(blob) % 4 != 0:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


class JsonMemoryStore:
    """Helpers for reading and writing memory JSON files under data_root/memory."""

    def __init__(self, data_root: str | Path | None = None) -> None:
        self.data_root = self.resolve_data_root(data_root)
        self.memory_dir = self.data_root / "memory"

    @staticmethod
    def resolve_data_root(data_root: str | Path | None = None) -> Path:
        if data_root is not None:
            return Path(data_root)
        env_path = os.getenv("BLAIRE_DATA_PATH")
        if env_path:
            return Path(env_path)
        return Path("./data")

    def ensure_memory_namespace(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for filename, default in _MEMORY_FILES.items():
            path = self.memory_dir / filename
            if not path.exists():
                self._atomic_write_json(path, default)

    def _load_json(self, filename: str, default: Any) -> Any:
        self.ensure_memory_namespace()
        path = self.memory_dir / filename
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                path.replace(backup_path)
            except OSError:
                logger.exception("memory_store: failed to backup corrupt file", extra={"path": str(path)})
            logger.error("memory_store: corrupt file recreated", extra={"path": str(path), "error": str(exc)})
            self._atomic_write_json(path, default)
            return default

    def _save_json(self, filename: str, payload: Any) -> None:
        self.ensure_memory_namespace()
        self._atomic_write_json(self.memory_dir / filename, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load_projects(self) -> list[dict[str, Any]]:
        return self._load_json("projects.json", [])

    def save_projects(self, projects: list[dict[str, Any]]) -> None:
        self._save_json("projects.json", projects)

    def load_todos(self) -> list[dict[str, Any]]:
        return self._load_json("todo.json", [])

    def save_todos(self, todos: list[dict[str, Any]]) -> None:
        self._save_json("todo.json", todos)

    def load_debates(self) -> list[dict[str, Any]]:
        return self._load_json("debates.json", [])

    def save_debates(self, debates: list[dict[str, Any]]) -> None:
        self._save_json("debates.json", debates)

    def load_predictions(self) -> list[dict[str, Any]]:
        return self._load_json("predictions.json", [])

    def save_predictions(self, predictions: list[dict[str, Any]]) -> None:
        self._save_json("predictions.json", predictions)

    def load_patterns(self) -> list[dict[str, Any]]:
        data = self._load_json("patterns.json", [])
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []

    def save_patterns(self, patterns: list[dict[str, Any]]) -> None:
        self._save_json("patterns.json", patterns)


class StructuredMemoryStore:
    """SQLite-backed structured memory store under data_root/memory."""

    def __init__(self, data_root: str | Path | None = None) -> None:
        self.data_root = JsonMemoryStore.resolve_data_root(data_root)
        self.memory_dir = self.data_root / "memory"
        self.db_path = self.memory_dir / "blaire_memory.db"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row[1]) for row in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def initialize(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    summarised_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    stability TEXT NOT NULL,
                    embedding BLOB
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source_window TEXT,
                    tags TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    importance INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "memories", "text_hash", "text_hash TEXT")
            self._ensure_column(conn, "patterns", "text_hash", "text_hash TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_seen ON memories(last_seen_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_updated ON patterns(updated_at)")
            rows = conn.execute("SELECT id, text FROM memories WHERE (text_hash IS NULL OR text_hash = '')").fetchall()
            for row in rows:
                conn.execute("UPDATE memories SET text_hash = ? WHERE id = ?", (_normalized_text_hash(str(row[1])), int(row[0])))
            conn.execute(
                """
                DELETE FROM memories
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM memories
                    GROUP BY type, text_hash
                )
                """
            )
            rows = conn.execute("SELECT id, text FROM patterns WHERE (text_hash IS NULL OR text_hash = '')").fetchall()
            for row in rows:
                conn.execute("UPDATE patterns SET text_hash = ? WHERE id = ?", (_normalized_text_hash(str(row[1])), int(row[0])))
            conn.execute(
                """
                DELETE FROM patterns
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM patterns
                    GROUP BY text_hash
                )
                """
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_type_hash ON memories(type, text_hash)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_patterns_hash ON patterns(text_hash)")
            conn.commit()

    def log_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        session_id: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        self.initialize()
        ts = timestamp or _iso_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=True)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events(timestamp, session_id, event_type, payload_json, summarised_at) VALUES (?, ?, ?, ?, NULL)",
                (ts, session_id, event_type.strip() or "unknown", payload_json),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def add_or_update_memory(
        self,
        *,
        memory_type: str,
        text: str,
        tags: list[str] | str | None = None,
        importance: int = 3,
        stability: str = "evolving",
        now: str | None = None,
    ) -> int:
        self.initialize()
        normalized_type = memory_type.strip().lower()
        if normalized_type not in _MEMORY_TYPES:
            normalized_type = "fact"
        normalized_stability = stability.strip().lower()
        if normalized_stability not in _STABILITY_VALUES:
            normalized_stability = "evolving"
        trimmed_text = " ".join(text.split()).strip()[:512]
        if not trimmed_text:
            raise ValueError("memory text cannot be empty")
        safe_importance = max(1, min(5, int(importance)))
        ts = now or _iso_now()
        normalized_tags = _normalize_tags(tags)
        embedding_blob = _embedding_to_blob(embed_text(trimmed_text))
        text_hash = _normalized_text_hash(trimmed_text)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, tags, importance, stability FROM memories WHERE type = ? AND text_hash = ? ORDER BY id LIMIT 1",
                (normalized_type, text_hash),
            ).fetchone()
            if existing:
                existing_id, existing_tags_raw, existing_importance, existing_stability = existing
                merged_tags = _normalize_tags(_safe_json_list(existing_tags_raw))
                merged_tags = _normalize_tags([*merged_tags, *normalized_tags])
                merged_importance = max(int(existing_importance), safe_importance)
                merged_stability = existing_stability if existing_stability == "stable" else normalized_stability
                conn.execute(
                    """
                    UPDATE memories
                    SET tags = ?, last_seen_at = ?, importance = ?, stability = ?, embedding = ?
                    WHERE id = ?
                    """,
                    (json.dumps(merged_tags), ts, merged_importance, merged_stability, embedding_blob, existing_id),
                )
                conn.commit()
                return int(existing_id)
            cursor = conn.execute(
                """
                INSERT INTO memories(type, text, tags, created_at, last_seen_at, importance, stability, embedding, text_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_type,
                    trimmed_text,
                    json.dumps(normalized_tags),
                    ts,
                    ts,
                    safe_importance,
                    normalized_stability,
                    embedding_blob,
                    text_hash,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def add_or_update_memories(self, items: list[dict[str, Any]]) -> int:
        self.initialize()
        inserted_or_updated = 0
        with self._connect() as conn:
            for item in items:
                memory_type = str(item.get("memory_type", item.get("type", "fact")))
                text = str(item.get("text", ""))
                tags = item.get("tags", [])
                importance = int(item.get("importance", 3))
                stability = str(item.get("stability", "evolving"))
                now = str(item.get("now", _iso_now()))
                normalized_type = memory_type.strip().lower()
                if normalized_type not in _MEMORY_TYPES:
                    normalized_type = "fact"
                normalized_stability = stability.strip().lower()
                if normalized_stability not in _STABILITY_VALUES:
                    normalized_stability = "evolving"
                trimmed_text = " ".join(text.split()).strip()[:512]
                if not trimmed_text:
                    continue
                safe_importance = max(1, min(5, int(importance)))
                normalized_tags = _normalize_tags(tags if isinstance(tags, (list, tuple, str)) else [])
                embedding_blob = _embedding_to_blob(embed_text(trimmed_text))
                text_hash = _normalized_text_hash(trimmed_text)
                existing = conn.execute(
                    "SELECT id, tags, importance, stability FROM memories WHERE type = ? AND text_hash = ? ORDER BY id LIMIT 1",
                    (normalized_type, text_hash),
                ).fetchone()
                if existing:
                    existing_id, existing_tags_raw, existing_importance, existing_stability = existing
                    merged_tags = _normalize_tags(_safe_json_list(existing_tags_raw))
                    merged_tags = _normalize_tags([*merged_tags, *normalized_tags])
                    merged_importance = max(int(existing_importance), safe_importance)
                    merged_stability = existing_stability if existing_stability == "stable" else normalized_stability
                    conn.execute(
                        """
                        UPDATE memories
                        SET tags = ?, last_seen_at = ?, importance = ?, stability = ?, embedding = ?
                        WHERE id = ?
                        """,
                        (json.dumps(merged_tags), now, merged_importance, merged_stability, embedding_blob, existing_id),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO memories(type, text, tags, created_at, last_seen_at, importance, stability, embedding, text_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            normalized_type,
                            trimmed_text,
                            json.dumps(normalized_tags),
                            now,
                            now,
                            safe_importance,
                            normalized_stability,
                            embedding_blob,
                            text_hash,
                        ),
                    )
                inserted_or_updated += 1
            conn.commit()
        return inserted_or_updated

    def get_memories(
        self,
        *,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self.initialize()
        max_limit = max(1, min(500, int(limit)))
        query = "SELECT id, type, text, tags, created_at, last_seen_at, importance, stability FROM memories"
        clauses: list[str] = []
        params: list[Any] = []
        if memory_type:
            clauses.append("type = ?")
            params.append(memory_type.strip().lower())
        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"
        query = f"{query} ORDER BY importance DESC, last_seen_at DESC LIMIT ?"
        params.append(max_limit)
        out: list[dict[str, Any]] = []
        with self._connect() as conn:
            for row in conn.execute(query, params).fetchall():
                row_tags = _normalize_tags(_safe_json_list(row[3]))
                if tags and not any(tag.lower() in {t.lower() for t in row_tags} for tag in tags):
                    continue
                out.append(
                    {
                        "id": int(row[0]),
                        "type": str(row[1]),
                        "text": str(row[2]),
                        "tags": row_tags,
                        "created_at": str(row[4]),
                        "last_seen_at": str(row[5]),
                        "importance": int(row[6]),
                        "stability": str(row[7]),
                    }
                )
        return out

    def add_or_update_pattern(
        self,
        *,
        text: str,
        source_window: str,
        tags: list[str] | str | None = None,
        importance: int = 3,
        updated_at: str | None = None,
    ) -> int:
        self.initialize()
        cleaned_text = " ".join(text.split()).strip()[:512]
        if not cleaned_text:
            raise ValueError("pattern text cannot be empty")
        safe_importance = max(1, min(5, int(importance)))
        ts = updated_at or _iso_now()
        normalized_tags = _normalize_tags(tags)
        text_hash = _normalized_text_hash(cleaned_text)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, tags, importance FROM patterns WHERE text_hash = ? ORDER BY id LIMIT 1",
                (text_hash,),
            ).fetchone()
            if existing:
                existing_id, existing_tags_raw, existing_importance = existing
                merged_tags = _normalize_tags(_safe_json_list(existing_tags_raw))
                merged_tags = _normalize_tags([*merged_tags, *normalized_tags])
                merged_importance = max(int(existing_importance), safe_importance)
                conn.execute(
                    "UPDATE patterns SET source_window = ?, tags = ?, updated_at = ?, importance = ? WHERE id = ?",
                    (source_window, json.dumps(merged_tags), ts, merged_importance, existing_id),
                )
                conn.commit()
                return int(existing_id)
            cursor = conn.execute(
                "INSERT INTO patterns(text, source_window, tags, updated_at, importance, text_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (cleaned_text, source_window, json.dumps(normalized_tags), ts, safe_importance, text_hash),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def add_or_update_patterns(self, items: list[dict[str, Any]], source_window: str) -> int:
        self.initialize()
        count = 0
        with self._connect() as conn:
            for item in items:
                cleaned_text = " ".join(str(item.get("text", "")).split()).strip()[:512]
                if not cleaned_text:
                    continue
                safe_importance = max(1, min(5, int(item.get("importance", 3))))
                ts = str(item.get("updated_at", _iso_now()))
                normalized_tags = _normalize_tags(item.get("tags", []))
                text_hash = _normalized_text_hash(cleaned_text)
                existing = conn.execute(
                    "SELECT id, tags, importance FROM patterns WHERE text_hash = ? ORDER BY id LIMIT 1",
                    (text_hash,),
                ).fetchone()
                if existing:
                    existing_id, existing_tags_raw, existing_importance = existing
                    merged_tags = _normalize_tags(_safe_json_list(existing_tags_raw))
                    merged_tags = _normalize_tags([*merged_tags, *normalized_tags])
                    merged_importance = max(int(existing_importance), safe_importance)
                    conn.execute(
                        "UPDATE patterns SET source_window = ?, tags = ?, updated_at = ?, importance = ? WHERE id = ?",
                        (source_window, json.dumps(merged_tags), ts, merged_importance, existing_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO patterns(text, source_window, tags, updated_at, importance, text_hash) VALUES (?, ?, ?, ?, ?, ?)",
                        (cleaned_text, source_window, json.dumps(normalized_tags), ts, safe_importance, text_hash),
                    )
                count += 1
            conn.commit()
        return count

    def get_top_patterns(self, limit: int = 10, tags: list[str] | None = None) -> list[dict[str, Any]]:
        self.initialize()
        max_limit = max(1, min(100, int(limit)))
        out: list[dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, text, source_window, tags, updated_at, importance FROM patterns ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (max_limit * 2,),
            ).fetchall()
        for row in rows:
            row_tags = _normalize_tags(_safe_json_list(row[3]))
            if tags and not any(tag.lower() in {t.lower() for t in row_tags} for tag in tags):
                continue
            out.append(
                {
                    "id": int(row[0]),
                    "text": str(row[1]),
                    "source_window": str(row[2] or ""),
                    "tags": row_tags,
                    "updated_at": str(row[4]),
                    "importance": int(row[5]),
                }
            )
            if len(out) >= max_limit:
                break
        return out

    def retrieve_relevant_memories(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        self.initialize()
        clean_query = " ".join(query.split()).strip()
        if not clean_query:
            return []
        query_embedding = embed_text(clean_query)
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        with self._connect() as conn:
            for row in conn.execute(
                "SELECT id, type, text, tags, created_at, last_seen_at, importance, stability, embedding FROM memories"
            ).fetchall():
                memory_embedding = _blob_to_embedding(row[8])
                if not memory_embedding:
                    continue
                if len(memory_embedding) != len(query_embedding):
                    continue
                dot_score = sum(a * b for a, b in zip(query_embedding, memory_embedding))
                recency_days = max(0.0, (now - _load_iso(str(row[5])).astimezone(timezone.utc)).total_seconds() / 86400.0)
                recency_score = 1.0 / (1.0 + recency_days / 30.0)
                importance_score = max(1, min(5, int(row[6]))) / 5.0
                score = (0.65 * dot_score) + (0.25 * importance_score) + (0.10 * recency_score)
                rows.append(
                    {
                        "id": int(row[0]),
                        "type": str(row[1]),
                        "text": str(row[2]),
                        "tags": _normalize_tags(_safe_json_list(row[3])),
                        "created_at": str(row[4]),
                        "last_seen_at": str(row[5]),
                        "importance": int(row[6]),
                        "stability": str(row[7]),
                        "score": score,
                    }
                )
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows[: max(1, min(100, int(limit)))]

    def list_recent_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.get_memories(limit=limit)

    def get_stats(self) -> dict[str, int]:
        self.initialize()
        with self._connect() as conn:
            events_count = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            memories_count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            patterns_count = int(conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0])
        return {"events": events_count, "memories": memories_count, "patterns": patterns_count}

    def get_events_since(self, since_iso: str, limit: int = 1000, unsummarised_only: bool = False) -> list[dict[str, Any]]:
        self.initialize()
        where_clause = "WHERE julianday(timestamp) >= julianday(?)"
        if unsummarised_only:
            where_clause = f"{where_clause} AND (summarised_at IS NULL OR summarised_at = '')"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, session_id, event_type, payload_json, summarised_at
                FROM events
                """
                + where_clause
                + """
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (since_iso, max(1, min(5000, int(limit)))),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload: dict[str, Any] = {}
            try:
                raw = json.loads(str(row[4] or "{}"))
                if isinstance(raw, dict):
                    payload = raw
            except json.JSONDecodeError:
                pass
            out.append(
                {
                    "id": int(row[0]),
                    "timestamp": str(row[1]),
                    "session_id": str(row[2] or ""),
                    "event_type": str(row[3]),
                    "payload": payload,
                    "summarised_at": str(row[5] or ""),
                }
            )
        return out

    def mark_events_summarised(self, event_ids: list[int], summarised_at: str | None = None) -> None:
        self.initialize()
        if not event_ids:
            return
        ts = summarised_at or _iso_now()
        placeholders = ",".join("?" for _ in event_ids)
        values: list[Any] = [ts, *[int(event_id) for event_id in event_ids]]
        with self._connect() as conn:
            conn.execute(f"UPDATE events SET summarised_at = ? WHERE id IN ({placeholders})", values)
            conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            conn.commit()

    def get_meta(self, key: str) -> str | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return str(row[0])

    def prune_old_events(self, keep_days: int = 30) -> int:
        self.initialize()
        keep_days = max(1, int(keep_days))
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        cutoff_iso = cutoff.isoformat()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM events WHERE julianday(timestamp) < julianday(?)", (cutoff_iso,))
            deleted = int(cursor.rowcount if cursor.rowcount is not None else 0)
            conn.commit()
        return max(0, deleted)
