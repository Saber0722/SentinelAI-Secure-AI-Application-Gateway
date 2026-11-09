"""In-memory request limits for a publicly reachable gateway. Off unless configured."""

from __future__ import annotations

import threading
import time
from collections import deque

WINDOW_S = 60.0
_MAX_TRACKED_CLIENTS = 10_000


class SlidingWindowLimiter:
    """Allow at most ``limit`` requests per client in any 60-second window."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._hits: dict[str, deque[float]] = {}
        self._refused: set[str] = set()
        self._lock = threading.Lock()

    def check(self, client: str, now: float | None = None) -> tuple[bool, bool]:
        """Return ``(allowed, first_refusal)``. Only the first refusal of a burst should be audited,
        so that a flood of refused requests cannot fill the audit log."""
        now = time.monotonic() if now is None else now
        cutoff = now - WINDOW_S
        with self._lock:
            if len(self._hits) >= _MAX_TRACKED_CLIENTS:
                self._forget_idle(cutoff)
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) < self.limit:
                hits.append(now)
                self._refused.discard(client)
                return True, False
            first = client not in self._refused
            self._refused.add(client)
            return False, first

    def _forget_idle(self, cutoff: float) -> None:
        for client in [c for c, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[client]
            self._refused.discard(client)
