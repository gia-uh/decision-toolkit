"""Rate limiting and bounded-concurrency primitives for the REST server.

Both are process-local on purpose: decision-toolkit is a local-first, in-process
runtime. Multi-worker deployments get one limiter per worker; that limitation is
documented rather than hidden behind a network dependency.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    last_seen: float


class RateLimiter:
    """Token bucket per key (API-key hash or client IP).

    ``rate`` is the sustained number of requests per ``window`` seconds; the
    bucket capacity is ``burst`` (defaults to ``rate``). ``rate <= 0`` disables
    limiting.
    """

    def __init__(
        self,
        rate: int = 0,
        window: float = 60.0,
        *,
        burst: int | None = None,
    ) -> None:
        self.rate = max(0, int(rate))
        self.window = max(0.001, float(window))
        self.burst = max(1, int(burst if burst is not None else self.rate))
        self._refill = self.rate / self.window if self.rate else 0.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def allow(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)`` for ``key``."""
        if not self.enabled:
            return True, 0.0
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), last_seen=timestamp)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, timestamp - bucket.last_seen)
                bucket.tokens = min(
                    float(self.burst),
                    bucket.tokens + elapsed * self._refill,
                )
                bucket.last_seen = timestamp
            if len(self._buckets) >= 10_000:
                self._cleanup(timestamp)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            retry_after = (1.0 - bucket.tokens) / self._refill
            return False, max(0.0, retry_after)

    def _cleanup(self, now: float) -> None:
        if len(self._buckets) < 10_000:
            return
        cutoff = now - max(120.0, self.window * 2.0)
        stale = [key for key, bucket in self._buckets.items() if bucket.last_seen < cutoff]
        for key in stale:
            self._buckets.pop(key, None)


class ConcurrencyGate:
    """Async gate that rejects instead of building an unbounded queue."""

    def __init__(self, max_concurrent: int = 4) -> None:
        self.max_concurrent = max(0, int(max_concurrent))
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def enabled(self) -> bool:
        return self.max_concurrent > 0

    async def try_acquire(self) -> bool:
        if not self.enabled:
            return True
        async with self._lock:
            if self._active >= self.max_concurrent:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        if not self.enabled:
            return
        async with self._lock:
            self._active = max(0, self._active - 1)


__all__ = ["ConcurrencyGate", "RateLimiter"]
