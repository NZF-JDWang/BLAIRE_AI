from collections import defaultdict, deque
from dataclasses import dataclass
from time import time

from fastapi import HTTPException


@dataclass(frozen=True)
class RateLimitRule:
    max_requests: int
    window_seconds: int


class InMemoryRateLimiter:
    def __init__(self):
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, rule: RateLimitRule) -> None:
        now = time()
        bucket = self._buckets[key]
        cutoff = now - rule.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= rule.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket.append(now)


rate_limiter = InMemoryRateLimiter()

