"""Tool-result distillation into durable structured memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from blaire_core.memory.store import MemoryStore


MIN_CONFIDENCE = 0.8
DEDUP_SIMILARITY_THRESHOLD = 0.9


@dataclass(slots=True)
class DistilledMemory:
    memory_type: str
    text: str
    confidence: float
    importance: int
    stability: str
    tags: list[str]


def _normalize_text(value: str) -> str:
    collapsed = " ".join(str(value).split()).strip()
    return collapsed[:512]


def _token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2}


def _similarity(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if not union:
        return 0.0
    return intersection / union


def _is_high_value_summary(text: str) -> bool:
    normalized = _normalize_text(text)
    return len(normalized) >= 32 and len(_token_set(normalized)) >= 5


def _extract_candidates(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> list[DistilledMemory]:
    if not result.get("ok"):
        return []

    data = result.get("data")
    candidates: list[DistilledMemory] = []

    if tool_name == "web_search" and isinstance(data, dict):
        query = _normalize_text(str(data.get("query", "")))
        rows = data.get("results", []) if isinstance(data.get("results"), list) else []
        if query and rows:
            first = rows[0] if isinstance(rows[0], dict) else {}
            title = _normalize_text(str(first.get("title", "")))
            url = _normalize_text(str(first.get("url", "")))
            if title and url:
                candidates.append(
                    DistilledMemory(
                        memory_type="fact",
                        text=f"Web lookup for '{query}' returned top result '{title}' ({url}).",
                        confidence=0.84,
                        importance=4,
                        stability="evolving",
                        tags=["web", "lookup"],
                    )
                )

    if tool_name == "check_disk_space" and isinstance(data, dict):
        used_percent = float(data.get("used_percent", 0.0))
        path = _normalize_text(str(data.get("path", ".")))
        if path and used_percent >= 85.0:
            candidates.append(
                DistilledMemory(
                    memory_type="fact",
                    text=f"Disk utilization is high at {used_percent:.2f}% for path '{path}'.",
                    confidence=0.93,
                    importance=5,
                    stability="stable",
                    tags=["system", "disk"],
                )
            )

    if tool_name == "local_search" and isinstance(data, dict):
        query = _normalize_text(str(args.get("query", "")))
        rows = data.get("results", []) if isinstance(data.get("results"), list) else []
        if query and rows:
            candidates.append(
                DistilledMemory(
                    memory_type="fact",
                    text=f"Local search for '{query}' returned {len(rows)} relevant entries.",
                    confidence=0.8,
                    importance=3,
                    stability="temp",
                    tags=["local", "lookup"],
                )
            )

    return candidates


def distill_tool_result_to_memory(
    memory: MemoryStore,
    *,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    min_confidence: float = MIN_CONFIDENCE,
    dedup_similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Extract high-signal facts/preferences from tool responses and store them conservatively."""
    if not result.get("ok"):
        return {"written": 0, "skipped": 0, "reason": "tool_failed"}

    candidates = _extract_candidates(tool_name=tool_name, args=args, result=result)
    if not candidates:
        return {"written": 0, "skipped": 0, "reason": "no_candidates"}

    existing = memory.get_memories(tags=[f"tool:{tool_name}"], limit=200)
    existing_texts = [str(row.get("text", "")) for row in existing]
    written = 0
    skipped = 0

    for item in candidates:
        normalized_text = _normalize_text(item.text)
        if item.confidence < min_confidence or not _is_high_value_summary(normalized_text):
            skipped += 1
            continue
        if any(_similarity(normalized_text, old_text) >= dedup_similarity_threshold for old_text in existing_texts):
            skipped += 1
            continue

        tags = [
            f"tool:{tool_name}",
            "source:external",
            f"confidence:{item.confidence:.2f}",
            *item.tags,
        ]
        memory.add_or_update_memory(
            memory_type=item.memory_type,
            text=normalized_text,
            tags=tags,
            importance=item.importance,
            stability=item.stability,
        )
        existing_texts.append(normalized_text)
        written += 1

    return {"written": written, "skipped": skipped, "reason": "ok"}
