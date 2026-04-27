import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            hit_window = self._hits[key]
            while hit_window and hit_window[0] < cutoff:
                hit_window.popleft()

            if len(hit_window) >= limit:
                retry_after = int(max(1, window_seconds - (now - hit_window[0])))
                return False, retry_after

            hit_window.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = InMemoryRateLimiter()
