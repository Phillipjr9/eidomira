"""Authentication: registration, sign-in, email tokens, and the guest/bearer paths.

This module had one test asserting a JWT subject while performing password hashing,
token issuance and access control. The tests below cover the paths that actually decide
who gets in — including the account-enumeration property in `authenticate`.
"""
from __future__ import annotations

import hashlib
import statistics
import time
import uuid

import jwt
import pytest
from argon2 import PasswordHasher
from fastapi import HTTPException

from app import security
from app.config import settings
from app.database import database
from app.security import access_token, hasher

PASSWORD = "correct horse battery staple"


def make_user(email: str, *, password_hash: str | None = None, disabled: int = 0,
              verified: bool = False, user_id: str | None = None) -> str:
    """Insert a user directly, so tests that are not about `register` skip the hashing."""
    user_id = user_id or uuid.uuid4().hex
    now = int(time.time())
    database.execute(
        "INSERT INTO users(id,email,password_hash,created_at,disabled,email_verified_at)"
        " VALUES(?,?,?,?,?,?)",
        (user_id, email, password_hash or hasher.hash(PASSWORD), now, disabled,
         now if verified else None),
    )
    return user_id


@pytest.fixture(scope="session")
def stored_hash():
    """One argon2 derivation reused across tests; each costs ~50 ms to compute."""
    return hasher.hash(PASSWORD)


class CountingHasher:
    """Wraps the real hasher to count verification work.

    `PasswordHasher.verify` cannot be patched directly ("object attribute 'verify' is
    read-only"), so the module attribute is replaced instead and everything else
    delegates through `__getattr__`.
    """

    def __init__(self, inner):
        self._inner = inner
        self.verifications = []

    def verify(self, stored, password):
        self.verifications.append(password)
        return self._inner.verify(stored, password)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def counting_hasher(monkeypatch):
    counter = CountingHasher(hasher)
    monkeypatch.setattr(security, "hasher", counter)
    return counter


def call_optional_user(authorization=None, cookie=None, token_header=None):
    """Call it the way FastAPI does.

    The `Header(default=None)` marker is not a real default: passing no argument leaves
    the framework's sentinel in place, which a truthiness check reads as a credential and
    the fail-closed `except Exception` then reports as a 401. FastAPI always injects a str
    or None, so the tests do too — and `_as_token` now refuses the sentinel as well, so
    this helper is belt as well as braces.
    """
    return security.optional_user(authorization=authorization,
                                 eidomira_token=token_header,
                                 eidomira_access_token=cookie)


# ─────────────────────────────── registration ───────────────────────────────

@pytest.mark.parametrize("bad_email", [
    "no-at-sign", "two@@at.com", "spaces in@example.com", "@example.com", "user@nodot",
])
def test_register_rejects_malformed_email(isolated_db, bad_email):
    with pytest.raises(ValueError):
        security.register(bad_email, PASSWORD)


def test_register_rejects_short_password(isolated_db):
    with pytest.raises(ValueError):
        security.register("person@example.com", "short")


def test_register_normalises_case_and_whitespace(isolated_db):
    user = security.register("  Person@Example.COM  ", PASSWORD)
    assert user["email"] == "person@example.com"
    assert database.one("SELECT id FROM users WHERE email=?", ("person@example.com",))["id"] == user["id"]


def test_register_does_not_report_a_taken_email_as_a_generic_failure(isolated_db):
    security.register("taken@example.com", PASSWORD)
    with pytest.raises(ValueError, match="already uses that email"):
        security.register("taken@example.com", PASSWORD)


def test_register_treats_a_taken_email_case_insensitively(isolated_db):
    """The column is COLLATE NOCASE, so the constraint must fire on case alone."""
    security.register("Case@Example.com", PASSWORD)
    with pytest.raises(ValueError, match="already uses that email"):
        security.register("case@example.com", PASSWORD)


def test_register_starts_unverified(isolated_db):
    assert security.register("fresh@example.com", PASSWORD)["email_verified"] is False


def test_register_stores_a_verifiable_hash_and_never_the_password(isolated_db):
    security.register("hashed@example.com", PASSWORD)
    row = database.one("SELECT password_hash FROM users WHERE email=?", ("hashed@example.com",))
    assert PASSWORD not in row["password_hash"]
    assert row["password_hash"].startswith("$argon2id$")
    assert hasher.verify(row["password_hash"], PASSWORD) is True


# ──────────────────────────────── sign-in ────────────────────────────────

def test_authenticate_accepts_the_right_password(isolated_db, stored_hash):
    uid = make_user("right@example.com", password_hash=stored_hash)
    user = security.authenticate("right@example.com", PASSWORD)
    assert user == {"id": uid, "email": "right@example.com", "email_verified": False,
                    "role": "user"}


def test_authenticate_is_case_insensitive_on_the_address(isolated_db, stored_hash):
    make_user("mixed@example.com", password_hash=stored_hash)
    assert security.authenticate("  MIXED@example.com ", PASSWORD) is not None


def test_authenticate_rejects_a_wrong_password(isolated_db, stored_hash):
    make_user("wrong@example.com", password_hash=stored_hash)
    assert security.authenticate("wrong@example.com", "not the password") is None


def test_authenticate_rejects_an_unknown_address(isolated_db, stored_hash):
    assert security.authenticate("nobody@example.com", PASSWORD) is None


def test_authenticate_reports_a_verified_address(isolated_db, stored_hash):
    make_user("verified@example.com", password_hash=stored_hash, verified=True)
    assert security.authenticate("verified@example.com", PASSWORD)["email_verified"] is True


def test_authenticate_refuses_a_disabled_account(isolated_db, stored_hash):
    make_user("disabled@example.com", password_hash=stored_hash, disabled=1)
    assert security.authenticate("disabled@example.com", PASSWORD) is None


def test_authenticate_upgrades_a_hash_made_with_weak_parameters(isolated_db):
    """A stored hash below the current cost must be silently re-hashed on sign-in."""
    weak = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)
    weak_hash = weak.hash(PASSWORD)
    make_user("stale@example.com", password_hash=weak_hash)

    assert security.authenticate("stale@example.com", PASSWORD) is not None
    upgraded = database.one("SELECT password_hash FROM users WHERE email=?", ("stale@example.com",))
    assert upgraded["password_hash"] != weak_hash
    assert hasher.check_needs_rehash(upgraded["password_hash"]) is False


def test_authenticate_leaves_a_current_hash_alone(isolated_db, stored_hash):
    make_user("current@example.com", password_hash=stored_hash)
    security.authenticate("current@example.com", PASSWORD)
    unchanged = database.one("SELECT password_hash FROM users WHERE email=?", ("current@example.com",))
    assert unchanged["password_hash"] == stored_hash


# ────────────── the enumeration property this module is here to guard ──────────────
#
# `authenticate` used to return before hashing anything when the address had no account,
# which made the response time a reliable oracle: 96 ms for a registered address against
# 2 ms for an unregistered one, both answering an identical 401.

def test_every_failed_sign_in_path_performs_identical_verification_work(
        isolated_db, counting_hasher, stored_hash):
    """The deterministic form of the property: same work whether or not an account exists.

    A registered address, a disabled one and an unregistered one must all cost exactly
    one password verification. Returning early for the unregistered case is what created
    the oracle.
    """
    make_user("known@example.com", password_hash=stored_hash)
    make_user("off@example.com", password_hash=stored_hash, disabled=1)

    assert security.authenticate("known@example.com", "wrong") is None
    assert security.authenticate("off@example.com", PASSWORD) is None
    assert security.authenticate("absent@example.com", "wrong") is None

    assert len(counting_hasher.verifications) == 3, (
        f"verification work differed between paths: {counting_hasher.verifications!r}"
    )


def test_a_correct_password_also_verifies_exactly_once(isolated_db, counting_hasher, stored_hash):
    make_user("right@example.com", password_hash=stored_hash)
    assert security.authenticate("right@example.com", PASSWORD) is not None
    assert len(counting_hasher.verifications) == 1


def test_no_account_existence_signal_in_response_time(isolated_db, stored_hash):
    """End-to-end timing check of the property above.

    The deterministic call-count tests are the real guard; this one confirms the work
    actually costs the same wall-clock time. The bound is deliberately loose — both
    branches run identical code, so the ratio sits near 1 whether the machine is idle
    or loaded, while the bug this replaces was a ~50x gap.
    """
    make_user("known@example.com", password_hash=stored_hash)

    def median_ms(address, runs=5):
        security.authenticate(address, "definitely wrong")  # warm
        samples = []
        for _ in range(runs):
            started = time.perf_counter()
            security.authenticate(address, "definitely wrong")
            samples.append((time.perf_counter() - started) * 1000)
        return statistics.median(samples)

    known, unknown = median_ms("known@example.com"), median_ms("absent@example.com")
    assert unknown >= known * 0.5, (
        f"response time distinguishes registered from unregistered addresses: "
        f"{known:.1f} ms vs {unknown:.1f} ms"
    )


# ───────────────────────────── email verification tokens ─────────────────────────────

def test_email_token_verifies_the_address_once(isolated_db, stored_hash):
    uid = make_user("token@example.com", password_hash=stored_hash)
    raw = security.issue_email_token(uid)

    user = security.verify_email_token(raw)
    assert user == {"id": uid, "email": "token@example.com", "email_verified": True}


def test_email_token_cannot_be_replayed(isolated_db, stored_hash):
    uid = make_user("once@example.com", password_hash=stored_hash)
    raw = security.issue_email_token(uid)

    assert security.verify_email_token(raw) is not None
    assert security.verify_email_token(raw) is None


def test_email_token_expiry_is_enforced(isolated_db, stored_hash):
    uid = make_user("expired@example.com", password_hash=stored_hash)
    raw = security.issue_email_token(uid)
    database.execute("UPDATE email_verification_tokens SET expires_at=? WHERE user_id=?",
                     (int(time.time()) - 1, uid))
    assert security.verify_email_token(raw) is None


def test_unknown_email_token_is_rejected(isolated_db):
    assert security.verify_email_token("not-a-real-token") is None


def test_issuing_a_new_token_invalidates_the_previous_one(isolated_db, stored_hash):
    uid = make_user("resend@example.com", password_hash=stored_hash)
    first = security.issue_email_token(uid)
    second = security.issue_email_token(uid)

    assert security.verify_email_token(first) is None
    assert security.verify_email_token(second) is not None


def test_tokens_are_stored_hashed(isolated_db, stored_hash):
    """A database leak must not hand over working verification links."""
    uid = make_user("hashed-token@example.com", password_hash=stored_hash)
    raw = security.issue_email_token(uid)
    rows = database.one("SELECT token_hash FROM email_verification_tokens WHERE user_id=?", (uid,))
    assert rows["token_hash"] == hashlib.sha256(raw.encode()).hexdigest()
    assert raw != rows["token_hash"]


# ──────────────────────── access tokens and access control ────────────────────────

def test_access_token_subject(isolated_db):
    token = access_token({"id": "user1", "email": "a@example.com"})
    payload = jwt.decode(token, settings.auth_secret, algorithms=["HS256"])
    assert payload["sub"] == "user1" and payload["typ"] == "access"


def test_access_token_is_bound_to_the_configured_secret(isolated_db):
    token = access_token({"id": "user1", "email": "a@example.com"})
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, "a-different-secret", algorithms=["HS256"])


def test_bearer_token_resolves_the_current_user(isolated_db, stored_hash):
    uid = make_user("bearer@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "bearer@example.com"})
    user = call_optional_user(authorization=f"Bearer {token}")
    assert user["id"] == uid


def test_cookie_is_accepted_as_an_alternative_to_the_header(isolated_db, stored_hash):
    uid = make_user("cookie@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "cookie@example.com"})
    assert call_optional_user(cookie=token)["id"] == uid


def test_bearer_scheme_is_required(isolated_db, stored_hash):
    uid = make_user("scheme@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "scheme@example.com"})
    with pytest.raises(HTTPException) as excinfo:
        security.optional_user(authorization=f"Basic {token}")
    assert excinfo.value.status_code == 401


def test_an_empty_authorization_header_counts_as_anonymous(isolated_db, monkeypatch):
    """Not an error: an empty header carries no credentials, so it takes the guest path."""
    monkeypatch.setattr(settings, "require_auth", False)
    assert call_optional_user(authorization="")["id"] == "local-guest"


def test_forged_and_garbage_tokens_are_rejected(isolated_db):
    # "Bearer" alone has no token to parse and must not fall through to the guest path.
    for header in ["Bearer nonsense", "Bearer", "Bearer a.b.c", "Bearer ..."]:
        with pytest.raises(HTTPException) as excinfo:
            call_optional_user(authorization=header)
        assert excinfo.value.status_code == 401


def test_token_for_a_deleted_user_is_rejected(isolated_db, stored_hash):
    uid = make_user("gone@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "gone@example.com"})
    database.execute("DELETE FROM users WHERE id=?", (uid,))
    with pytest.raises(HTTPException) as excinfo:
        call_optional_user(authorization=f"Bearer {token}")
    assert excinfo.value.status_code == 401


def test_disable_takes_effect_immediately_for_an_issued_token(isolated_db, stored_hash):
    """Access tokens carry no revocation list, so the per-request lookup is the control."""
    uid = make_user("revoked@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "revoked@example.com"})
    assert call_optional_user(authorization=f"Bearer {token}")["id"] == uid

    database.execute("UPDATE users SET disabled=1 WHERE id=?", (uid,))
    with pytest.raises(HTTPException) as excinfo:
        call_optional_user(authorization=f"Bearer {token}")
    assert excinfo.value.status_code == 401


def test_the_alternate_header_is_accepted_where_authorization_is_lost(isolated_db, stored_hash):
    """`Authorization` is the right header and remains the primary one, but the hosted
    preview has been seen to lose it in transit — every request carrying a credential in a
    body arrived intact while every request carrying one in that header arrived with no
    credential at all. So the same token also travels in a plain header, and either is enough.
    """
    uid = make_user("alternate@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "alternate@example.com"})
    assert call_optional_user(token_header=token)["id"] == uid


def test_a_refused_credential_is_not_retried_as_another(isolated_db, stored_hash):
    """Two ways in, one answer. A caller that sends a bad `Authorization` and a good
    alternate is not authenticated by the second: falling through on a refused token would
    turn it into a puzzle, and the answer is 401 either way."""
    uid = make_user("both@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "both@example.com"})
    with pytest.raises(HTTPException) as refused:
        call_optional_user(authorization="Bearer not-a-token", token_header=token)
    assert refused.value.status_code == 401


def test_unsigned_guest_access_when_auth_is_not_required(isolated_db, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", False)
    assert call_optional_user()["id"] == "local-guest"


def test_guest_access_is_refused_when_auth_is_required(isolated_db, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    with pytest.raises(HTTPException) as excinfo:
        call_optional_user()
    assert excinfo.value.status_code == 401


def test_authenticated_user_rejects_the_guest(isolated_db, monkeypatch):
    monkeypatch.setattr(settings, "require_auth", False)
    with pytest.raises(HTTPException) as excinfo:
        security.authenticated_user(authorization=None)
    assert excinfo.value.status_code == 401


def test_authenticated_user_accepts_a_real_token(isolated_db, stored_hash):
    uid = make_user("private@example.com", password_hash=stored_hash)
    token = access_token({"id": uid, "email": "private@example.com"})
    assert security.authenticated_user(authorization=f"Bearer {token}")["id"] == uid
