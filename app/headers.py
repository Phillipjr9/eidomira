from fastapi import Request

# Routes that are safe to embed. The public marketing page holds no session state and no
# credentials, so framing it (product embeds, hosted previews, docs) is allowed. Everything
# else — the private workspace and the API — keeps the strict anti-framing policy.
EMBEDDABLE_PATHS = {"/", "/index.html"}

# The on-device landing demo fetches a pinned MediaPipe build. `script-src` already allowed
# this CDN; the WASM runtime is fetched at run time, so `connect-src` and `worker-src` need it.
DEMO_CDN = "https://cdn.jsdelivr.net"


def _is_embeddable(path: str) -> bool:
    return path in EMBEDDABLE_PATHS or path.startswith("/static/")


async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)

    frame_policy = "*" if _is_embeddable(request.url.path) else "'self'"
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(self), microphone=(self), fullscreen=(self)",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Content-Security-Policy": (
            "default-src 'self'; "
            # 'wasm-unsafe-eval' permits WebAssembly compilation only (not JS eval); the
            # on-device tracking demo is a wasm build and Chrome refuses it without this.
            f"script-src 'self' {DEMO_CDN} 'wasm-unsafe-eval'; "
            "style-src 'self'; font-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            f"connect-src 'self' {DEMO_CDN} ws: wss:; worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'self'; "
            f"frame-ancestors {frame_policy}; form-action 'self'"
        ),
    })

    if not _is_embeddable(request.url.path):
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response
