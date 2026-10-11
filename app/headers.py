from fastapi import Request

from app.config import settings

# Routes that are safe to embed. The public marketing page holds no session state and no
# credentials, so framing it (product embeds, hosted previews, docs) is allowed. Everything
# else — the private workspace and the API — keeps the strict anti-framing policy.
EMBEDDABLE_PATHS = {"/", "/index.html"}

# Two on-device pages fetch from this CDN: the landing demo pulls a pinned MediaPipe build,
# and the lab at /lab pulls a pinned onnxruntime-web and its wasm binary. `script-src` allows
# the origin; the wasm is fetched at run time, so `connect-src` and `worker-src` need it too,
# and `wasm-unsafe-eval` is what lets the browser compile either of them.
DEMO_CDN = "https://cdn.jsdelivr.net"


def _is_embeddable(path: str) -> bool:
    return path in EMBEDDABLE_PATHS or path.startswith("/static/")


def _configured_ancestors() -> str | None:
    """What the operator allowed to frame the private pages, or None if nothing is allowed.

    Kept separate from `_is_embeddable`: the marketing page is framed by design, the studio is
    not, and a deployment that embeds the studio has said so on purpose.
    """
    allowed = [origin.strip() for origin in settings.embed_ancestors.replace(",", " ").split()]
    return " ".join(allowed) if allowed else None


async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)

    embeddable = _is_embeddable(request.url.path)
    ancestors = _configured_ancestors()
    frame_policy = "*" if embeddable else (ancestors or "'self'")
    permissions_policy = "camera=*, microphone=*, fullscreen=*" if (embeddable or ancestors) else "camera=(self), microphone=(self), fullscreen=(self)"
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": permissions_policy,
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

    # X-Frame-Options cannot express a list of ancestors, and browsers that understand it
    # ignore it when a frame-ancestors policy is present. It is sent only where it agrees with
    # the policy: for the private pages with nothing configured, where it is the belt to the
    # CSP's braces, and never for the pages an operator has chosen to embed.
    if not embeddable and ancestors is None:
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif response.headers.get("content-type", "").startswith("text/html") or \
            request.url.path.startswith("/static/"):
        # Revalidate rather than reuse. Without this a browser is entitled to serve a page or
        # its script from its own cache without asking, which is how a change that is live on
        # the server stays invisible in a tab that was already open. ETags still turn the
        # revalidation into a 304, so this costs a round trip, not a download.
        response.headers["Cache-Control"] = "no-cache"
    return response
