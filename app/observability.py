from __future__ import annotations

import logging
import time
from fastapi import Request
from prometheus_client import Counter, Histogram, Gauge

from app.security import TOKEN_HEADER

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
    path = request.url.path
    if not path.startswith("/api/"):
        return response

    sender = request.client.host if request.client else "unknown"
    logger = logging.getLogger("uvicorn.error")
    # Which of the ways a token can travel actually arrived. This is the reading that took two
    # sessions to obtain: the access log showed a signed-in browser being answered as a guest,
    # and nothing in it could say whether the credential had been refused or had never been
    # sent. It says which now, and by which route, because the two have different fixes.
    credential = (request.headers.get("authorization")
                  or request.headers.get(TOKEN_HEADER)
                  or request.cookies.get("eidomira_access_token"))
    route = ("authorization" if request.headers.get("authorization")
             else "x-eidomira-token" if request.headers.get(TOKEN_HEADER)
             else "cookie" if request.cookies.get("eidomira_access_token") else "none")

    if path == "/api/auth/me" and response.status_code == 200:
        # The line the incident needed. This endpoint answers 200 either way — with the
        # account, or with a guest record when nothing arrived — so the status alone never
        # said which, and "signed in" and "holding a token the server never saw" were the same
        # three characters in the log. It says which now.
        logger.info(
            "200 GET /api/auth/me from %s — %s", sender,
            "a credential arrived via %s and was accepted" % route if credential
            else "no credential arrived by any route, so this is the guest record",
        )
    elif response.status_code == 401:
        if path == "/api/auth/login":
            # Nothing anonymous happens here: the credential is in the body, and saying "no
            # credential was sent" beside a 401 from the sign-in form would be a lie that
            # sends the next reader after the wrong thing, as it nearly did.
            detail = "the submitted email and password matched no account"
        elif credential:
            detail = "a credential arrived via %s and was refused" % route
        else:
            detail = "no credential was sent (anonymous)"
        logger.info("401 %s %s from %s — %s", request.method, path, sender, detail)
    return response
