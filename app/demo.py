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
* an owner who wants a password they can *type* can set `STUDIO_DEMO_PASSWORD`, which makes
  the demo accounts accept it on the ordinary sign-in form and prints it on the card. That is
  the whole purpose of setting it, so it is only honoured while the demo is on, and it has to
  pass the same length rule as any other password;
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

from argon2.exceptions import InvalidHashError, VerificationError

from app.billing import create_trial
from app.config import settings
from app.database import database
from app.security import MINIMUM_PASSWORD, hasher, register as register_account

#: Reserved by RFC 2606 for exactly this: names that can never be real.
DOMAIN = "eidomira.test"
ACCOUNTS = {"user": f"demo@{DOMAIN}", "admin": f"admin@{DOMAIN}"}
ROLES = tuple(ACCOUNTS)

#: Fixed ids, not `uuid4`, and that is the point of them.
#:
#: A session is signed over the account's id. Those accounts are rebuilt whenever the database
#: is — a fresh deployment, a wiped volume, or the throwaway database a hosted preview runs on
#: — and with a random id each rebuild, every token minted before it pointed at a row that no
#: longer existed. The server answered 401, the page signed the visitor out, and the same thing
#: happened again at the next rebuild. Measured here, repeatedly: the sandbox's database does
#: not survive a restart, so this was a logout every time the preview restarted.
#:
#: With fixed ids the rebuilt row is the *same* account, the token still names it, and the
#: session survives. Nothing else changes: the ids are still opaque strings in the same column,
#: and an existing installation keeps whatever ids it already has, because this only decides
#: what a *new* demo account is created with.
DEMO_IDS = {"user": "demouser000000000000000000000000", "admin": "demoadmin00000000000000000000000"}

#: The password the demo accounts are created with when `STUDIO_DEMO_PASSWORD` is not set.
#:
#: It used to be a random string nobody could read, on the reasoning that a published password
#: is a real account anyone can walk into. The owner of this installation asked for the
#: opposite, twice: a password the demo actually uses, so the demo exercises the real sign-in
#: path rather than a private shortcut, and so somebody can type it if the buttons are not
#: there. That is the caller's call to make, and it is made safely rather than absolutely:
#: this is only reachable while demo sign-in is on, and demo sign-in refuses to run at all
#: while `PUBLIC_URL` is https. Set `STUDIO_DEMO_PASSWORD` to use a different one.
DEFAULT_PASSWORD = "eidomira-demo-2026"


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
    if settings.demo_password and len(settings.demo_password) < MINIMUM_PASSWORD:
        # Refused rather than truncated or padded: the accounts are created through the
        # ordinary registration path, which enforces this rule, and a demo account that
        # could not have been registered normally would demonstrate the wrong thing.
        return (
            f"STUDIO_DEMO_PASSWORD is shorter than {MINIMUM_PASSWORD} characters, which the "
            "ordinary sign-in rules refuse, so demo sign-in stays off. Use a longer password, "
            "or leave it unset to keep the random ones."
        )
    return None


def enabled() -> bool:
    return refusal_reason() is None and bool(settings.demo_login)


def is_demo_email(email: str | None) -> bool:
    return bool(email) and email.strip().lower().endswith("@" + DOMAIN)


def published_password() -> str | None:
    """The password the demo accounts accept, or None when demo sign-in is off.

    Always something while the demo is on — the configured value, else DEFAULT_PASSWORD — so
    the accounts always have a password and the card can always show it. None whenever the
    demo is off, so the API never hands out a credential for a feature that is not running.
    """
    if not enabled():
        return None
    return settings.demo_password or DEFAULT_PASSWORD


def _adopt_published_password(user_id: str, current_hash: str) -> None:
    """Bring an existing demo account's password in line with the published one.

    This is what makes `STUDIO_DEMO_PASSWORD` take effect on a restart rather than only for
    accounts created afterwards, and what gives an account seeded before this default existed
    a password that can actually be typed.
    """
    password = published_password()
    if not password:
        return
    try:
        if hasher.verify(current_hash, password):
            return
    except (VerificationError, InvalidHashError):
        pass
    database.execute("UPDATE users SET password_hash=? WHERE id=?", (hasher.hash(password), user_id))


def ensure_accounts() -> list[str]:
    """Create whichever demo accounts are missing. Returns the addresses it created.

    Called at boot only while `enabled()`, and idempotent, so restarting does not accumulate
    accounts or reset the credits somebody was looking at. Both accounts get the ordinary
    verified-email trial, because a demo account that behaves unlike a real one demonstrates
    the wrong thing.

    It also keeps existing demo accounts' passwords in step with the published one, so setting
    `STUDIO_DEMO_PASSWORD` takes effect on the next boot rather than only for accounts created
    afterwards.
    """
    created: list[str] = []
    for role, email in ACCOUNTS.items():
        # An account that already exists is left exactly as it is, id included: moving an id
        # would mean moving every wallet, subscription and ledger row that references it.
        existing = database.one("SELECT id,role,password_hash FROM users WHERE email=?", (email,))
        if existing:
            if role == "admin" and existing["role"] != "admin":
                database.execute("UPDATE users SET role='admin' WHERE id=?", (existing["id"],))
            _adopt_published_password(existing["id"], existing["password_hash"])
            continue
        # The published password, always: these accounts are meant to be signed into.
        user = register_account(email, published_password() or secrets.token_urlsafe(32),
                                user_id=DEMO_IDS[role])
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
