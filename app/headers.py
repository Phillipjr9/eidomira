from fastapi import Request


async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(self), microphone=(self), fullscreen=(self)",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "connect-src 'self' ws: wss:; object-src 'none'; base-uri 'self'; "
            "frame-ancestors 'self'; form-action 'self'"
        ),
    })
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response
