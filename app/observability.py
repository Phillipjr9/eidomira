from __future__ import annotations

import time
from fastapi import Request
from prometheus_client import Counter, Histogram, Gauge

REQUESTS = Counter("eidomira_http_requests_total", "HTTP requests", ["method","route","status"])
DURATION = Histogram("eidomira_http_duration_seconds", "HTTP duration", ["method","route"])
ACTIVE_SESSIONS = Gauge("eidomira_active_sessions", "In-memory identity sessions")
ACTIVE_PEERS = Gauge("eidomira_active_webrtc_peers", "Active WebRTC peers")


async def metrics_middleware(request: Request, call_next):
    started=time.perf_counter();response=None
    try:
        response=await call_next(request);return response
    finally:
        route=getattr(request.scope.get("route"),"path",request.url.path)
        status=str(response.status_code if response else 500)
        REQUESTS.labels(request.method,route,status).inc()
        DURATION.labels(request.method,route).observe(time.perf_counter()-started)
