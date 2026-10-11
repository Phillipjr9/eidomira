from __future__ import annotations

import re
import time
import uuid
import sqlite3
import secrets
import hashlib
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Header, HTTPException, Cookie, Depends

from app.config import settings
from app.database import database

EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
#: The one place the length rule lives. A configured demo password is checked against it
#: too: it reaches `register`, and a rule enforced in two places drifts.
MINIMUM_PASSWORD = 10
hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)

# A throwaway hash of a random secret, used to spend the same argon2 work when an
# address has no account to check against. Without it, authenticate() returned before
# hashing anything and the response time alone revealed whether an email was
# registered — measured at 96 ms for a known address versus 2 ms for an unknown one
# over HTTP, both answering an identical 401.
_ENUMERATION_DECOY = hasher.hash(secrets.token_urlsafe(32))


def _spend_verification(password: str) -> None:
    """Do the work of a failed password check when there is no stored hash to use.

    Deliberately discards the result: the only point is that the caller cannot be
    told apart from one that really did check a password.
    """
    try:
        hasher.verify(_ENUMERATION_DECOY, password)
    except VerifyMismatchError:
        pass


def register(email: str, password: str, user_id: str | None = None):
    """Create an account. `user_id` is normally left to chance.

    It is settable for the one caller that needs an account to have the *same* identity every
    time it is created: the demo accounts, whose row is rebuilt whenever the database is
    (app/demo.py explains why that matters). Everything else about this path — the address
    rules, the password length, the uniqueness — is identical either way, so a fixed id is not
    a way to make a special account.
    """
    email = email.strip().lower()
    if not EMAIL.fullmatch(email): raise ValueError("Enter a valid email address")
    if len(password) < MINIMUM_PASSWORD:
        raise ValueError(f"Password must contain at least {MINIMUM_PASSWORD} characters")
    user_id = user_id or uuid.uuid4().hex
    try:
        database.execute("INSERT INTO users(id,email,password_hash,created_at) VALUES(?,?,?,?)",
                         (user_id, email, hasher.hash(password), int(time.time())))
    except sqlite3.IntegrityError as exc:
        # Constraint name, not a bare substring: a NOT NULL or foreign-key failure must
        # not be reported to the caller as "that email is taken".
        if "users.email" in str(exc): raise ValueError("An account already uses that email") from exc
        raise
    # Not read from the request: whatever a caller sends, a new account is an ordinary
    # account. Promotion is an operator action (tools/grant_admin.py), never a parameter.
    return {"id": user_id, "email": email, "email_verified": False, "role": "user"}


def issue_email_token(user_id: str):
    raw=secrets.token_urlsafe(32); digest=hashlib.sha256(raw.encode()).hexdigest(); created=int(time.time())
    database.execute("UPDATE email_verification_tokens SET used_at=? WHERE user_id=? AND used_at IS NULL",(created,user_id))
    database.execute("INSERT INTO email_verification_tokens(id,user_id,token_hash,expires_at,created_at) VALUES(?,?,?,?,?)",
                     (uuid.uuid4().hex,user_id,digest,created+settings.email_token_ttl,created))
    return raw


def verify_email_token(raw: str):
    digest=hashlib.sha256(raw.encode()).hexdigest(); current=int(time.time())
    token=database.one("SELECT * FROM email_verification_tokens WHERE token_hash=? AND used_at IS NULL",(digest,))
    if not token or token["expires_at"] < current: return None
    with database.lock, database.connect() as db:
        db.execute("UPDATE email_verification_tokens SET used_at=? WHERE id=? AND used_at IS NULL",(current,token["id"]))
        db.execute("UPDATE users SET email_verified_at=COALESCE(email_verified_at,?) WHERE id=?",(current,token["user_id"]))
        user=db.execute("SELECT id,email,email_verified_at FROM users WHERE id=?",(token["user_id"],)).fetchone()
    return {"id":user[0],"email":user[1],"email_verified":bool(user[2])}


def authenticate(email: str, password: str):
    user = database.one("SELECT * FROM users WHERE email=? AND disabled=0", (email.strip().lower(),))
    if not user:
        # Unregistered address, or a disabled account: both are filtered out by the query
        # above, and both must still cost one verification so the timing does not
        # distinguish them from a wrong password. See _ENUMERATION_DECOY.
        _spend_verification(password)
        return None
    try: hasher.verify(user["password_hash"], password)
    except VerifyMismatchError: return None
    if hasher.check_needs_rehash(user["password_hash"]):
        database.execute("UPDATE users SET password_hash=? WHERE id=?", (hasher.hash(password), user["id"]))
    return {"id": user["id"], "email": user["email"], "email_verified":bool(user.get("email_verified_at")),
            "role": user.get("role") or "user"}


def change_password(user_id: str, current_password: str, new_password: str) -> None:
    """Replace one account's password, having proved the current one.

    The current password is required and verified rather than merely present in the session:
    a token left on an unlocked laptop should not be enough to lock its owner out of their
    own account. Everything else follows the rules the sign-up form already applies, since a
    password that could not have been registered should not become reachable by changing to
    it.

    Sessions that are already open are deliberately not revoked. They are signed tokens with
    nothing to revoke, and pretending otherwise would be worse than saying so: the Settings
    page tells the account holder exactly that, and the token expires on its own.
    """
    user = database.one("SELECT password_hash FROM users WHERE id=? AND disabled=0", (user_id,))
    if not user:
        raise LookupError("Authentication required")
    try:
        hasher.verify(user["password_hash"], current_password)
    except VerifyMismatchError:
        # PermissionError, not a 401: the caller *is* authenticated, and this one action is
        # refused. The distinction is load-bearing on the client, where a 401 on a request
        # that carried a token means the session itself is over — and a mistyped password is
        # not that.
        raise PermissionError("That is not your current password") from None
    if len(new_password) < MINIMUM_PASSWORD:
        raise ValueError(f"Password must contain at least {MINIMUM_PASSWORD} characters")
    if new_password == current_password:
        raise ValueError("Choose a password different from the current one")
    database.execute("UPDATE users SET password_hash=? WHERE id=?",
                     (hasher.hash(new_password), user_id))


def access_token(user):
    now = int(time.time())
    return jwt.encode({"sub":user["id"],"email":user["email"],"iat":now,"nbf":now-2,
                       "exp":now+settings.access_token_ttl,"typ":"access"},
                      settings.auth_secret, algorithm="HS256")


#: A header a client may present the token in instead of `Authorization`.
#:
#: This exists because of what the hosted preview proved about `Authorization`: it does not
#: always survive the trip. Every request whose credential travelled in the *body* arrived
#: intact — `POST /api/auth/login` and `POST /api/auth/demo-login` both answered 200 — while
#: every request carrying a token in `Authorization` reached this server with no credential at
#: all. `/api/auth/me` answered as a guest to a browser that was holding a token, and
#: `/api/billing/account` answered 401, and minting a brand-new token changed neither, because
#: the new one went missing the same way. The cookie cannot cover for it: it is `SameSite=Lax`,
#: and a browser does not send that from inside a cross-site iframe, which an embedded preview
#: is. A request that reaches the application is a request that reached it; the header is how
#: the caller says who they are, and a client that can say it in two ways should.
TOKEN_HEADER = "X-Eidomira-Token"


def _as_token(value) -> str | None:
    """Only a string is a credential.

    Called directly rather than through FastAPI — which is what a test does — the parameters
    keep their `Header(default=None)` sentinels, and a sentinel is an object, so it is truthy.
    A truthiness check would hand one to the decoder and answer a perfectly anonymous call with
    401. FastAPI injects a `str` or `None`; everything else is nothing.
    """
    return value if isinstance(value, str) else None


def _presented_token(authorization: str | None, alternate: str | None, cookie: str | None) -> str | None:
    """Whichever credential the caller offered — and only one of them.

    `Authorization` wins if it is there, because a caller that sends both means that one. A
    rejected credential is never retried as another: falling through on failure would turn a
    token this server has refused into a puzzle, and the answer is 401 either way.
    """
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "Invalid or expired access token")
        return token
    return alternate or cookie


def optional_user(authorization: str | None = Header(default=None),
                  eidomira_token: str | None = Header(default=None, alias=TOKEN_HEADER),
                  eidomira_access_token: str | None = Cookie(default=None)):
    token = _presented_token(_as_token(authorization), _as_token(eidomira_token),
                             _as_token(eidomira_access_token))
    if not token:
        if settings.require_auth: raise HTTPException(401, "Authentication required")
        return {"id":"local-guest","email":"local@eidomira.invalid","email_verified":False,"role":"guest"}
    try:
        payload = jwt.decode(token, settings.auth_secret, algorithms=["HS256"])
        # The role is read from the database on every request rather than carried in the
        # token. A token that claims a role it was not issued with therefore does nothing,
        # and demoting an administrator takes effect on their next click instead of when
        # their token expires.
        user = database.one("SELECT id,email,disabled,email_verified_at,role FROM users WHERE id=?", (payload["sub"],))
        if not user or user["disabled"]: raise ValueError
        return {"id":user["id"],"email":user["email"],"email_verified":bool(user["email_verified_at"]),
                "role":user["role"] or "user"}
    except Exception as exc:
        raise HTTPException(401, "Invalid or expired access token") from exc


def authenticated_user(authorization: str | None = Header(default=None),
                       eidomira_token: str | None = Header(default=None, alias=TOKEN_HEADER),
                       eidomira_access_token: str | None = Cookie(default=None)):
    user=optional_user(authorization,eidomira_token,eidomira_access_token)
    if user["id"] == "local-guest": raise HTTPException(401,"Authentication required")
    return user


def require_admin(user=Depends(authenticated_user)):
    """Owner access, for the console that reports what this installation really is.

    Deliberately stricter than "is signed in": an ordinary account gets 403, an anonymous
    caller gets 401, and the local-guest fallback (which exists so the studio can run
    without accounts) can never satisfy it.
    """
    if user.get("role") != "admin":
        raise HTTPException(403, "Administrator access required")
    return user
