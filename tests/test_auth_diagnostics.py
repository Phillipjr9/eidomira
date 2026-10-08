"""Two kinds of 401, and the one the client must not read as a logout.

This file exists because of a real incident and the way it was investigated. The user reported
being signed out; the access log of their browser's session showed `/api/auth/me` answering 200
and `/api/billing/account` answering 401 four times, in that order, and nothing beside either
line saying *why* it was refused. The two refusals mean opposite things:

  * "no credential was sent at all" — the caller is anonymous. Refusing is correct, and it says
    nothing about whether the session these pages care about is alive.
  * "a credential was sent and refused" — the token really is finished.

Both are 401, so the log could not settle which had happened, and neither could the browser:
`optional_user` returns a *guest record* for the first case, and `/api/auth/me` answers it with
200, so a client checking `response.ok` saw a signed-in user called local@eidomira.invalid.

The fix is a contract rather than a patch: only the endpoint that identifies a token may end a
session, it must be asked before anything is ended, and the server now says on the access log
line which kind of refusal it just sent.
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app import demo
from app.config import settings
from app.main import app

#: The address the guest record carries. Its whole purpose is to be recognisable as "nobody":
#: the studio rendered it into the account chip, which is how a page came to look signed in as
#: somebody who does not exist.
GUEST_ADDRESS = "local@eidomira.invalid"


@pytest.fixture
def signed_in(monkeypatch, isolated_db):
    """The demo on, so there is a real account and a real token to contrast with the guest."""
    monkeypatch.setattr(settings, "demo_login", True)
    monkeypatch.setattr(settings, "public_url", "http://127.0.0.1:8000")
    demo.ensure_accounts()
    return demo


# ── what the client has to be able to tell apart ────────────────────────────────

def test_the_session_endpoint_answers_an_anonymous_caller_with_a_guest(isolated_db):
    """200, and that is the trap. The page's only question was "did it work", which this
    answers yes to for a visitor who is not signed in at all."""
    with TestClient(app) as client:
        response = client.get("/api/auth/me")

    assert response.status_code == 200, "if this ever changes, the client check goes with it"
    guest = response.json()
    assert guest["role"] == "guest"
    assert guest["email"] == GUEST_ADDRESS


def test_the_same_endpoint_refuses_a_token_it_cannot_verify(isolated_db):
    """The other 401 — and this one *is* about the session."""
    with TestClient(app) as client:
        response = client.get("/api/auth/me", headers={"authorization": "Bearer not-a-token"})

    assert response.status_code == 401


def test_the_endpoint_that_started_this_is_only_a_refusal(isolated_db):
    """`/api/billing/account` answers 401 to an anonymous caller, by design. A signed-in
    visitor never sees it — the client proves that below — so reading *this* 401 as the
    session being over had nothing to gain and signed people out."""
    with TestClient(app) as client:
        assert client.get("/api/billing/account").status_code == 401


def test_a_real_token_gets_the_balance(signed_in):
    """The contrast that makes the whole diagnosis possible: same endpoint, same second, one
    401 and one 200 — so the 401 was never evidence about the session."""
    with TestClient(app) as client:
        token = client.post("/api/auth/demo-login", json={"role": "user"}).json()["access_token"]
        headers = {"authorization": f"Bearer {token}"}

        assert client.get("/api/auth/me", headers=headers).json()["role"] == "user"
        balance = client.get("/api/billing/account", headers=headers)

    assert balance.status_code == 200
    assert "wallet" in balance.json()


# ── and the log now says which one it was ───────────────────────────────────────

def test_an_anonymous_refusal_is_logged_as_anonymous(isolated_db, caplog):
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(app) as client:
            client.get("/api/billing/account")

    line = next((r.getMessage() for r in caplog.records
                 if "401 GET /api/billing/account" in (r.getMessage())), None)
    assert line, "the refusal was logged without saying anything about the credential"
    assert "no credential was sent (anonymous)" in line


def test_a_rejected_token_is_logged_as_a_rejected_token(isolated_db, caplog):
    """The distinction the incident could not make. This is the line that would have answered
    the question in one reading instead of an afternoon of inference."""
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(app) as client:
            client.get("/api/auth/me", headers={"authorization": "Bearer not-a-token"})

    line = next((r.getMessage() for r in caplog.records
                 if "401 GET /api/auth/me" in (r.getMessage())), None)
    assert line
    assert "a credential was sent and refused" in line


def test_the_session_endpoint_says_whether_anything_arrived(isolated_db, caplog):
    """The line the incident needed. This endpoint answers 200 either way — with the account,
    or with a guest record when no credential arrived — so in the log "signed in" and "holding
    a token the server never saw" were the same three characters. It says which now."""
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(app) as client:
            client.get("/api/auth/me")

    line = next((r.getMessage() for r in caplog.records if "200 GET /api/auth/me" in r.getMessage()), None)
    assert line, "a guest 200 is still indistinguishable from a signed-in one"
    assert "no credential arrived" in line
    assert "guest record" in line


def test_the_session_endpoint_says_when_a_credential_was_accepted(signed_in, caplog):
    with TestClient(app) as client:
        token = client.post("/api/auth/demo-login", json={"role": "user"}).json()["access_token"]
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            client.get("/api/auth/me", headers={"authorization": f"Bearer {token}"})

    line = next((r.getMessage() for r in caplog.records if "200 GET /api/auth/me" in r.getMessage()), None)
    assert line and "sent and accepted" in line


def test_a_refused_sign_in_does_not_claim_the_caller_was_anonymous(isolated_db, caplog):
    """A 401 from the sign-in form has nothing to do with an Authorization header — the
    credential is in the body. The generic wording said "no credential was sent (anonymous)"
    beside it, which is a lie, and it nearly sent this diagnosis after a missing header that
    was never supposed to be there."""
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(app) as client:
            response = client.post("/api/auth/login",
                                   json={"email": "demo@eidomira.test", "password": "not-the-one"})

    assert response.status_code == 401
    line = next((r.getMessage() for r in caplog.records if "401 POST /api/auth/login" in r.getMessage()), None)
    assert line
    assert "matched no account" in line
    assert "anonymous" not in line


def test_a_successful_call_says_nothing(caplog, signed_in):
    """The log has to stay worth reading: only refusals are mentioned, and only on /api/."""
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(app) as client:
            token = client.post("/api/auth/demo-login", json={"role": "user"}).json()["access_token"]
            client.get("/api/billing/account", headers={"authorization": f"Bearer {token}"})

    assert not [r for r in caplog.records if "401" in (r.getMessage())]
