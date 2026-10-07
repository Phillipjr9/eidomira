"""The owner's console: what this installation *is*, rather than what it promises.

Read-only, deliberately. Suspending an account or refunding a payment is a second product
with a second threat model, and this one exists to answer the questions an owner is
otherwise guessing at: who is registered, what has been paid, and whether this server can
actually do the thing the site advertises.

That last question is the reason the module exists. On this machine `models/` has never
held swap weights and no ONNX runtime has ever been installed, so the diagnostic engine is
not a degraded mode of a working one — it is the only engine that has run. A console that
printed "engine: up" without saying so would be worse than no console, because the owner
would believe the product was live. So `system()` computes a verdict and the reasons for
it, and the page shows the reasons.

Nothing here returns a password hash, a token, or a signing key. Account addresses are
shown in full: that is what the console is for, and an administrator who cannot see them
cannot help anyone. `tests/test_admin.py` holds both halves of that line.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from app import billing
from app.config import DEVELOPMENT_AUTH_SECRET, settings
from app.database import database

ROOT = Path(__file__).resolve().parent.parent

#: Importable runtimes that decide whether any neural model can load at all.
RUNTIMES = ("onnxruntime", "insightface", "torch")

#: Weight file types that would make the swap real. Nothing downloads these; they are
#: either present in the repository or they are not.
WEIGHT_SUFFIXES = (".onnx", ".pth", ".safetensors", ".bin", ".ckpt")

ACCOUNT_LIMIT = 200
PAYMENT_LIMIT = 25
REPORT_LIMIT = 20


def _rows(sql: str, values=()) -> list[dict]:
    with database.lock, database.connect() as db:
        return [dict(row) for row in db.execute(sql, values).fetchall()]


def _installed(module: str) -> bool:
    """Whether a module can be imported, without importing it.

    `find_spec` runs the finders rather than the module, so this does not pay the
    several-second cost of loading torch just to answer a question about it.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _weights(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
            found.append({"name": path.name, "bytes": path.stat().st_size,
                          "modified": int(path.stat().st_mtime)})
    return found


def system(engine) -> dict:
    """What this server can and cannot do, with the blockers spelled out.

    The verdict is computed from evidence rather than configured, because the failure this
    guards against is a deployment whose owner believes it is swapping faces.
    """
    runtimes = {name: _installed(name) for name in RUNTIMES}
    model_path = Path(settings.model_path)
    swap_weights = _weights(model_path.parent)
    configured_swap_present = model_path.is_file()
    parser_present = Path(settings.parser_model_path).is_file()
    restoration_present = Path(settings.restoration_model_path).is_file()

    blockers: list[str] = []
    if not configured_swap_present:
        blockers.append(
            f"No swap weights at {settings.model_path} — the neural swap has nothing to load."
        )
    if not runtimes["onnxruntime"]:
        blockers.append(
            "onnxruntime is not installed, so no ONNX model can be loaded even if it is present."
        )
    if not (runtimes["insightface"] or runtimes["torch"]):
        blockers.append(
            "Neither insightface nor torch is installed, so the face alignment a swap needs "
            "cannot be computed."
        )

    return {
        "engine": {
            "name": engine.name,
            "provider": getattr(engine, "provider", None),
            "providers": list(getattr(engine, "providers", ()) or ()),
            "accelerated": bool(getattr(engine, "accelerated", False)),
        },
        "can_swap": not blockers,
        "blockers": blockers,
        "runtimes": runtimes,
        "weights": {
            "swap": swap_weights,
            "configured_swap_present": configured_swap_present,
            "parser_present": parser_present,
            "restoration_present": restoration_present,
            "directory": str(model_path.parent),
        },
        "settings": {
            "backend": settings.backend,
            "swap_pixel_boost": settings.swap_pixel_boost,
            "restoration_model_path": str(settings.restoration_model_path),
            "restoration_visibility": settings.restoration_visibility,
            "tone_transfer_strength": settings.tone_transfer_strength,
            "parser_model_path": str(settings.parser_model_path),
            "temporal_strength": settings.temporal_strength,
            "max_frame_width": settings.max_frame_width,
            "max_active_peers": settings.max_active_peers,
            "max_sessions": settings.max_sessions,
            "session_ttl_seconds": settings.session_ttl_seconds,
            "trainer_enabled": settings.trainer_enabled,
            "trainer_report_dir": str(settings.trainer_report_dir),
            "database_path": str(settings.database_path),
            "public_url": settings.public_url,
        },
        "security": {
            # A boolean, never the value. An administrator needs to know the signing key is
            # the one from the repository without the console becoming the place it leaks.
            "auth_secret_is_repository_default": settings.auth_secret == DEVELOPMENT_AUTH_SECRET,
            "require_auth": settings.require_auth,
            "require_self_verification": settings.require_self_verification,
            "access_token_ttl": settings.access_token_ttl,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "login_limit_per_hour": settings.login_limit_per_hour,
            "enrollment_limit_per_hour": settings.enrollment_limit_per_hour,
            "email_configured": bool(settings.smtp_host),
            "paystack_configured": bool(settings.paystack_secret_key),
            "turn_configured": bool(settings.turn_urls and settings.turn_secret),
            "calls_configured": bool(settings.livekit_api_key and settings.livekit_api_secret),
        },
    }


def accounts() -> dict:
    """Registered accounts with what each one owes and is owed.

    Addresses and roles only. The password hash is never selected — not filtered later,
    not selected at all — so there is no line of code here that could return it.
    """
    total = _rows("SELECT COUNT(*) AS n FROM users")[0]["n"]
    rows = _rows(
        "SELECT id,email,created_at,email_verified_at,disabled,role FROM users "
        "ORDER BY created_at DESC LIMIT ?", (ACCOUNT_LIMIT,),
    )
    now = int(time.time())
    listed = []
    for row in rows:
        billing_account = billing.account(row["id"])
        subscription = billing_account["subscription"] or {}
        listed.append({
            "email": row["email"],
            "created_at": row["created_at"],
            "verified": bool(row["email_verified_at"]),
            "disabled": bool(row["disabled"]),
            "role": row["role"] or "user",
            "plan": subscription.get("plan"),
            "plan_status": subscription.get("status"),
            "plan_expires": subscription.get("current_period_end"),
            "plan_active": bool(subscription.get("current_period_end", 0) > now
                                and subscription.get("status") in {"trialing", "active"}),
            "credits": billing_account["wallet"]["total"],
            "credits_used": billing_account["credits_used"],
        })

    return {
        "total": total,
        "shown": len(listed),
        "truncated": total > len(listed),
        "limit": ACCOUNT_LIMIT,
        "accounts": listed,
        "counts": {
            "admins": sum(1 for a in listed if a["role"] == "admin"),
            "verified": sum(1 for a in listed if a["verified"]),
            "unverified": sum(1 for a in listed if not a["verified"]),
            "disabled": sum(1 for a in listed if a["disabled"]),
            "paying": sum(1 for a in listed if a["plan_active"]),
        },
    }


def payments() -> dict:
    """Paystack activity: the intents, and what actually settled.

    Amounts stay in kobo, which is the unit they are stored in. Converting for display is
    the page's job; converting here would put a money-shaped bug one round() away from the
    ledger.
    """
    totals = _rows(
        "SELECT status, COUNT(*) AS n, COALESCE(SUM(amount_kobo),0) AS kobo "
        "FROM payment_intents GROUP BY status"
    )
    by_status = {row["status"]: {"count": row["n"], "kobo": row["kobo"]} for row in totals}
    recent = _rows(
        "SELECT p.reference,p.kind,p.product,p.amount_kobo,p.status,p.created_at,p.completed_at,"
        "       u.email AS email "
        "FROM payment_intents p LEFT JOIN users u ON u.id = p.user_id "
        "ORDER BY p.created_at DESC LIMIT ?", (PAYMENT_LIMIT,),
    )
    events = _rows("SELECT COUNT(*) AS n FROM payment_events")[0]["n"]
    webhooks_unprocessed = _rows(
        "SELECT COUNT(*) AS n FROM payment_events WHERE processed_at IS NULL"
    )[0]["n"]
    return {
        "by_status": by_status,
        "recent": recent,
        "events_received": events,
        "webhooks_unprocessed": webhooks_unprocessed,
    }


def reports() -> dict:
    """The trainer's written reports, newest first.

    Listed rather than summarised: the numbers inside them are for a human reading a
    session, and re-parsing a report to show it in a table would invent a second source of
    truth for the same measurement.
    """
    directory = Path(settings.trainer_report_dir)
    if not directory.is_dir():
        return {"directory": str(directory), "count": 0, "latest": [], "truncated": False}
    files = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix == ".md"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    listed = [{"name": p.name, "bytes": p.stat().st_size, "modified": int(p.stat().st_mtime)}
              for p in files[:REPORT_LIMIT]]
    return {"directory": str(directory), "count": len(files), "latest": listed,
            "truncated": len(files) > len(listed)}


def overview(engine, sessions, peers) -> dict:
    """Everything the console shows, in one request.

    One call rather than one per panel: the whole payload is a few kilobytes on an
    installation of this size, and a console that half-loads is a console that lies.
    """
    return {
        "generated_at": int(time.time()),
        "accounts": accounts(),
        "payments": payments(),
        "reports": reports(),
        "system": system(engine),
        "live": {
            "sessions": len(getattr(sessions, "data", {})),
            "session_capacity": settings.max_sessions,
            "peers": len(getattr(peers, "peers", peers) or ()),
            "peer_capacity": settings.max_active_peers,
        },
    }
