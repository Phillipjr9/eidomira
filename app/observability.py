from __future__ import annotations

import logging
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


async def auth_diagnostics_middleware(request: Request, call_next):
    """Say why an API call was refused, on the line the refusal is logged.

    A 401 means two different things in this service and the access log could not tell them
    apart: "there was no token here" and "there was a token and it was rejected". One endpoint
    answering the first kind is what signed a signed-in visitor out of the studio, and the log
    of the incident — `401, 401, 401` with nothing beside them — could not settle which had
    happened, so the diagnosis went to the client and had to be inferred. It should not have
    been a guess. The caller's address and the path say enough to identify the account; the
    token is never logged, and only refusals on /api/ are mentioned.
    """
    response = await call_next(request)
    if response.status_code == 401 and request.url.path.startswith("/api/"):
        sent = request.headers.get("authorization") or request.cookies.get("eidomira_access_token")
        logging.getLogger("uvicorn.error").info(
            "401 %s %s from %s — %s",
            request.method, request.url.path,
            request.client.host if request.client else "unknown",
            "a credential was sent and refused" if sent else "no credential was sent (anonymous)",
        )
    return response
