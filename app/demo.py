"""A way in that exists only when somebody asks for it, and never with a published password.

One-click demo sign-in: the login card offers two buttons, the server mints a session for a
seeded account, and no credential is written into any file.

Putting demo credentials on a login page is how a demonstration turns into a back door. The
page is public, so the credentials are public with it, and the account they open is a real
account in the real database, holding whatever the real database holds. It is also the kind
of convenience that gets deployed by accident, and the failure is silent: the site looks
fine, and whoever reads the page is its administrator.

So:

* the demo is a flag — `STUDIO_DEMO_LOGIN`, off unless set;
* it is refused outright while `PUBLIC_URL` is https, because that is the combination
  nobody means to ship, and a warning would still leave the door open;
* the passwords are random and appear nowhere, not in a page and not in this file, because
  the one-click path does not need one — it mints a session, which is the thing actually
  being asked for;
* the addresses are in the reserved `.test` domain (RFC 2606), so they can never collide
  with a customer's address or receive real mail;
* and the owner console marks them, so an administrator does not mistake a demo account for
  somebody who signed up.

Turning the flag off stops the buttons. It does not delete the accounts — nothing in this
module deletes anything — so the console badge is what tells you they are still there.
"""
from __future__ import annotations

import secrets
import time

from app.billing import create_trial
from app.config import settings
from app.database import database
from app.security import register as register_account

#: Reserved by RFC 2606 for exactly this: names that can never be real.
DOMAIN = "eidomira.test"
ACCOUNTS = {"user": f"demo@{DOMAIN}", "admin": f"admin@{DOMAIN}"}
ROLES = tuple(ACCOUNTS)


def refusal_reason() -> str | None:
    """Why demo sign-in is off, when somebody asked for it and did not get it."""
    if not settings.demo_login:
        return None
    if settings.public_url.startswith("https://"):
        return (
            "STUDIO_DEMO_LOGIN is set but PUBLIC_URL is https, so one-click demo sign-in "
            "stays off: on a public page it would hand out an administrator account to "
            "anyone who loads it. Use real accounts, or run the demo on a local URL."
        )
    return None


def enabled() -> bool:
    return refusal_reason() is None and bool(settings.demo_login)


def is_demo_email(email: str | None) -> bool:
    return bool(email) and email.strip().lower().endswith("@" + DOMAIN)


def ensure_accounts() -> list[str]:
    """Create whichever demo accounts are missing. Returns the addresses it created.

    Called at boot only while `enabled()`, and idempotent, so restarting does not accumulate
    accounts or reset the credits somebody was looking at. Both accounts get the ordinary
    verified-email trial, because a demo account that behaves unlike a real one demonstrates
    the wrong thing.
    """
    created: list[str] = []
    for role, email in ACCOUNTS.items():
        existing = database.one("SELECT id,role FROM users WHERE email=?", (email,))
        if existing:
            if role == "admin" and existing["role"] != "admin":
                database.execute("UPDATE users SET role='admin' WHERE id=?", (existing["id"],))
            continue
        # The password is a random string nobody is ever shown, so password sign-in to a
        # demo account is not possible even for someone who knows the address.
        user = register_account(email, secrets.token_urlsafe(32))
        database.execute("UPDATE users SET email_verified_at=?, role=? WHERE id=?",
                         (int(time.time()), role, user["id"]))
        create_trial(user["id"])
        created.append(email)
    return created


def sign_in(role: str) -> dict:
    """The account behind a demo button, in the shape the rest of the app expects.

    The flag is checked before the role, and that order is the point: an installation
    without demo sign-in must answer 404 to every request, including one naming a role
    that does not exist. Validating the role first would answer 400 to a probe and tell
    it the route is there but switched off, which is the one thing the 404 hides.
    """
    if not enabled():
        raise PermissionError("Demo sign-in is not enabled")
    if role not in ACCOUNTS:
        raise ValueError("Unknown demo role")
    user = database.one(
        "SELECT id,email,email_verified_at,role,disabled FROM users WHERE email=?",
        (ACCOUNTS[role],),
    )
    if not user or user["disabled"]:
        raise LookupError("The demo account is missing or disabled")
    return {"id": user["id"], "email": user["email"],
            "email_verified": bool(user["email_verified_at"]), "role": user["role"] or "user"}
