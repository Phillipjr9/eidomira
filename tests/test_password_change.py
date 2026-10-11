"""Changing a password from the Settings page.

The studio could create accounts and sign them in, but an account whose password could not be
changed was an account its owner could not recover control of. This is the operation, and the
decisions in it are the ones worth pinning:

* the current password is required, because a session token alone should not let somebody
  lock the owner out of their own account;
* the new one obeys the same length rule as sign-up, since a password that could not have been
  registered should not become reachable by changing to it;
* open sessions are *not* revoked, and the page says so instead of implying otherwise. They
  are signed tokens with nothing to revoke.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import authenticate

PASSWORD = "correct-horse-battery"
NEW_PASSWORD = "a-different-passphrase"


@pytest.fixture(autouse=True)
def _always_isolated(isolated_db):
    yield


@pytest.fixture
def client(isolated_db):
    with TestClient(app) as client:
        yield client


def member(client, email="member@example.com", password=PASSWORD):
    assert client.post("/api/auth/register",
                       json={"email": email, "password": password}).status_code == 200
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    client.headers["authorization"] = f"Bearer {login.json()['access_token']}"
    return email


def change(client, current=PASSWORD, new=NEW_PASSWORD):
    return client.post("/api/auth/password",
                       json={"current_password": current, "new_password": new})


def test_an_anonymous_caller_cannot_change_anything(client):
    response = client.post("/api/auth/password",
                           json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})
    assert response.status_code == 401


def test_the_current_password_is_required(client):
    """A token is not proof of ownership: it is a bearer credential, and one left in a browser
    must not be enough to take the account away from its owner.

    Refused with 403, not 401, and the distinction is deliberate. 401 means "you are not
    authenticated", and the studio's client signs a user out when a request that carried a
    token is answered 401 — so a mistyped password would have logged them out of the page they
    were using to fix it. 403 says what is true: authenticated, and this one action refused.
    """
    email = member(client)

    response = change(client, current="not-the-password")

    assert response.status_code == 403
    assert "current password" in response.json()["error"]
    assert authenticate(email, PASSWORD), "the old password still works, as it should"
    assert authenticate(email, NEW_PASSWORD) is None


def test_a_failed_change_does_not_end_the_session(client):
    """The regression, stated the way it happened: the form is on the settings page, the
    session must survive being told the current password is wrong."""
    email = member(client)

    assert change(client, current="not-the-password").status_code == 403

    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/auth/me").json()["email"] == email


def test_a_session_for_a_deleted_account_is_a_401_not_a_403(client):
    """The other side of the same distinction: when the account really is gone, the session is
    unusable and 401 is the honest answer — that is the case the client signs out for."""
    import uuid as uuid_module

    from app.security import access_token
    orphan = access_token({"id": uuid_module.uuid4().hex, "email": "gone@example.com"})

    response = client.post("/api/auth/password", headers={"authorization": f"Bearer {orphan}"},
                           json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})

    assert response.status_code == 401


def test_the_new_password_obeys_the_sign_up_rule(client):
    email = member(client)

    response = change(client, new="short")

    assert response.status_code == 400
    assert "10" in response.json()["error"]
    assert authenticate(email, PASSWORD)


def test_choosing_the_same_password_again_is_refused(client):
    email = member(client)

    response = change(client, new=PASSWORD)

    assert response.status_code == 400
    assert authenticate(email, PASSWORD)


def test_the_password_actually_changes(client):
    email = member(client)

    assert change(client).status_code == 200

    assert authenticate(email, NEW_PASSWORD), "the new password must work"
    assert authenticate(email, PASSWORD) is None, "the old password must not"
    login = client.post("/api/auth/login", json={"email": email, "password": NEW_PASSWORD})
    assert login.status_code == 200


def test_the_answer_carries_no_hash(client):
    member(client)

    body = change(client).json()

    assert body == {"ok": True, "message": "Password updated"}
    assert "argon2" not in str(body) and "$" not in str(body)


def test_sessions_that_are_already_open_keep_working(client):
    """Documented behaviour rather than a happy accident: the Settings page tells the account
    holder that other devices stay signed in, so this pins what that sentence promises."""
    email = member(client)

    assert change(client).status_code == 200

    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/auth/me").json()["email"] == email


def test_a_disabled_account_cannot_change_its_password(client):
    email = member(client)
    from app.database import database
    database.execute("UPDATE users SET disabled=1 WHERE email=?", (email,))

    assert change(client).status_code == 401
