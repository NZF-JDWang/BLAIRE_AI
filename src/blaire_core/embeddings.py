"""Lightweight local text embeddings for memory retrieval."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_EMBED_DIM = 384
_MODEL_CACHE: Any | None = None


def _hash_embed_text(text: str) -> list[float]:
    tokens = _TOKEN_RE.findall(text.lower())
    vector = [0.0] * _EMBED_DIM
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "little") % _EMBED_DIM
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        weight = 1.0 + (digest[3] / 255.0)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def _sentence_transformers_embed(text: str) -> list[float] | None:
    global _MODEL_CACHE  # noqa: PLW0603
    model_name = os.getenv("BLAIRE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        if _MODEL_CACHE is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _MODEL_CACHE = SentenceTransformer(model_name)
        vector = _MODEL_CACHE.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in vector]
    except Exception:  # noqa: BLE001
        return None


def embed_text(text: str) -> list[float]:
    """Embed text using configured provider; always falls back to deterministic hash embedding."""
    provider = os.getenv("BLAIRE_EMBEDDING_PROVIDER", "local").strip().lower()
    if provider in {"local", "sentence-transformers", "st"}:
        vector = _sentence_transformers_embed(text)
        if vector:
            return vector
    return _hash_embed_text(text)
