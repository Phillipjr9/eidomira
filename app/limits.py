from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import settings


class SlidingWindowLimiter:
    def __init__(self):
        self.events = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str, limit: int, window: int):
        now = time.monotonic(); cutoff = now - window
        with self.lock:
            bucket = self.events[key]
            while bucket and bucket[0] < cutoff: bucket.popleft()
            if len(bucket) >= limit: return False, max(1, int(window - (now - bucket[0])))
            bucket.append(now)
            if len(self.events) > 10000:
                for old in [k for k,v in self.events.items() if not v or v[-1] < cutoff][:1000]: self.events.pop(old,None)
            return True, 0


limiter = SlidingWindowLimiter()


async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith(("/static/", "/metrics")):
        return await call_next(request)
    client = request.client.host if request.client else "unknown"
    limit, window = settings.rate_limit_per_minute, 60
    if request.url.path == "/api/sessions" and request.method == "POST":
        limit, window = settings.enrollment_limit_per_hour, 3600
    allowed, retry = limiter.allow(f"{client}:{request.url.path}:{request.method}", limit, window)
    if not allowed:
        return JSONResponse({"error":"Rate limit exceeded"}, status_code=429,
                            headers={"Retry-After":str(retry)})
    return await call_next(request)
