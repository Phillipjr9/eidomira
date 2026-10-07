from __future__ import annotations

import re
import time
import uuid
import jwt

from app.config import settings

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]")


def clean(value: str, limit: int = 48) -> str:
    return SAFE_NAME.sub("-", value.strip())[:limit].strip("-")


def create_room_name() -> str:
    return f"{settings.call_room_prefix}-{uuid.uuid4().hex[:12]}"


def create_call_token(room: str, display_name: str, can_publish: bool = True) -> dict:
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise RuntimeError("LiveKit credentials are not configured")
    room = clean(room)
    name = clean(display_name) or "guest"
    if not room:
        raise ValueError("Invalid room name")
    now = int(time.time())
    identity = f"{name}-{uuid.uuid4().hex[:8]}"
    payload = {
        "iss": settings.livekit_api_key,
        "sub": identity,
        "name": name,
        "nbf": now - 5,
        "exp": now + settings.call_token_ttl,
        "video": {
            "roomJoin": True,
            "room": room,
            "canPublish": can_publish,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    token = jwt.encode(payload, settings.livekit_api_secret, algorithm="HS256")
    return {"token": token, "url": settings.livekit_url, "room": room,
            "identity": identity, "expires_in": settings.call_token_ttl}
