"""Sign-in and registration as the client actually sees them.

These go through the HTTP routes rather than the functions, because the per-account
throttle lives in `app/main.py` and nothing exercised it. The limiter is a module-level
singleton shared by every request, so each test clears it — otherwise budget spent by one
test would decide another test's outcome.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.limits import limiter
from app.main import app

PASSWORD = "a-sufficiently-long-passphrase"


@pytest.fixture
def client(isolated_db):
    limiter.events.clear()
    with TestClient(app) as test_client:
        yield test_client
    limiter.events.clear()


def sign_up(client, email: str, password: str = PASSWORD):
    return client.post("/api/auth/register", json={"email": email, "password": password})


def sign_in(client, email: str, password: str):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def test_registration_then_sign_in(client):
    assert sign_up(client, "route@example.com").status_code == 200
    response = sign_in(client, "route@example.com", PASSWORD)
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert response.cookies.get("eidomira_access_token")


def test_wrong_password_does_not_say_which_part_was_wrong(client):
    sign_up(client, "generic@example.com")
    response = sign_in(client, "generic@example.com", "wrong")
    assert response.status_code == 401
    assert response.json()["error"] == "Invalid email or password"


def test_unknown_address_gives_the_identical_answer(client):
    sign_up(client, "known@example.com")
    wrong = sign_in(client, "known@example.com", "wrong")
    unknown = sign_in(client, "absent@example.com", "wrong")
    assert (wrong.status_code, wrong.json()) == (unknown.status_code, unknown.json())


def test_token_from_sign_in_authenticates_other_routes(client):
    sign_up(client, "session@example.com")
    token = sign_in(client, "session@example.com", PASSWORD).json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["email"] == "session@example.com"


def test_failed_sign_ins_are_throttled_per_account(client, monkeypatch):
    monkeypatch.setattr(settings, "login_limit_per_hour", 3)
    sign_up(client, "throttled@example.com")

    for _ in range(3):
        assert sign_in(client, "throttled@example.com", "wrong").status_code == 401

    blocked = sign_in(client, "throttled@example.com", "wrong")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_successful_sign_ins_do_not_consume_the_failure_budget(client, monkeypatch):
    """Nobody throttles themselves by using the product normally."""
    monkeypatch.setattr(settings, "login_limit_per_hour", 2)
    sign_up(client, "regular@example.com")

    for _ in range(5):
        assert sign_in(client, "regular@example.com", PASSWORD).status_code == 200

    # five sign-ins spent nothing: the full failure budget is still available
    assert sign_in(client, "regular@example.com", "wrong").status_code == 401
    assert sign_in(client, "regular@example.com", "wrong").status_code == 401


def test_a_lockout_refuses_even_the_correct_password(client, monkeypatch):
    """The deliberate cost of a per-account lockout, recorded so it is not a surprise.

    The budget is checked before the password is, which is the only ordering that
    actually slows guessing down. The consequence is that a locked-out account stays
    locked out for the window even when the caller has the right password — and that an
    attacker who knows an address can trigger that lockout. Bounded by
    `login_limit_per_hour` and the one-hour window.
    """
    monkeypatch.setattr(settings, "login_limit_per_hour", 2)
    sign_up(client, "locked@example.com")

    for _ in range(2):
        sign_in(client, "locked@example.com", "wrong")
    assert sign_in(client, "locked@example.com", PASSWORD).status_code == 429


def test_throttling_one_account_leaves_others_alone(client, monkeypatch):
    monkeypatch.setattr(settings, "login_limit_per_hour", 2)
    sign_up(client, "victim@example.com")
    sign_up(client, "bystander@example.com")

    for _ in range(3):
        sign_in(client, "victim@example.com", "wrong")
    assert sign_in(client, "victim@example.com", PASSWORD).status_code == 429

    assert sign_in(client, "bystander@example.com", PASSWORD).status_code == 200


def test_throttle_key_does_not_expose_the_address(client):
    """The limiter holds digests, so dumped state carries no email addresses."""
    sign_in(client, "private.person@example.com", "wrong")
    keys = " ".join(limiter.events.keys())
    assert "private.person" not in keys and "example.com" not in keys


def test_registration_rejects_a_weak_password_over_http(client):
    response = sign_up(client, "weak@example.com", "short")
    assert response.status_code == 400
    assert "10 characters" in response.json()["error"]


def test_registration_reports_a_taken_email(client):
    sign_up(client, "taken-route@example.com")
    response = sign_up(client, "taken-route@example.com")
    assert response.status_code == 400
    assert "already uses that email" in response.json()["error"]


def test_returned_user_object_never_contains_the_password_hash(client):
    sign_up(client, "nohash@example.com")
    body = sign_in(client, "nohash@example.com", PASSWORD).json()
    assert "password" not in str(body).lower() or "password_hash" not in str(body)
    assert set(body["user"]) == {"id", "email", "email_verified"}
