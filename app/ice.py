from __future__ import annotations

import base64
import hashlib
import hmac
import time

from app.config import settings


def rtc_configuration(user_id: str) -> dict:
    """Return browser ICE configuration with ephemeral TURN REST credentials."""
    servers: list[dict] = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if settings.turn_url_list and settings.turn_secret:
        expiry = int(time.time()) + settings.turn_credential_ttl
        username = f"{expiry}:{user_id}"
        digest = hmac.new(
            settings.turn_secret.encode(), username.encode(), hashlib.sha1
        ).digest()
        credential = base64.b64encode(digest).decode()
        servers.append({
            "urls": settings.turn_url_list,
            "username": username,
            "credential": credential,
            "credentialType": "password",
        })
    return {"iceServers": servers, "iceTransportPolicy": "all",
            "credential_expires_in": settings.turn_credential_ttl}
