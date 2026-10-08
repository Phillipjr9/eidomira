"""Two kinds of 401, and the one the client must not read as a logout.

This file exists because of a real incident and the way it was investigated. The user reported
being signed out; the access log of their browser's session showed `/api/auth/me` answering 200
and `/api/billing/account` answering 401 four times, in that order, and nothing beside either
line saying *why* it was refused. The two refusals mean opposite things:

  * "no credential was sent at all" — the caller is anonymous. Refusing is correct, and it says
    nothing about whether the session these pages care about is alive.
  * "a credential arrived via authorization and was refused" — the token really is finished.

Both are 401, so the log could not settle which had happened, and neither could the browser:
`optional_user` returns a *guest record* for the first case, and `/api/auth/me` answers it with
200, so a client checking `response.ok` saw a signed-in user called local@eidomira.invalid.

The fix is a contract rather than a patch: only the endpoint that identifies a token may end a
session, it must be asked before anything is ended, and the server now says on the access log
line which kind of refusal it just sent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import demo
from app.config import settings
from app.main import app

ROOT = Path(__file__).resolve().parent.parent

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


# ── the second way in, and why it exists ────────────────────────────────────────
#
# The hosted preview loses `Authorization`. Every request whose credential travelled in a body
# arrived intact — `POST /api/auth/login` and `POST /api/auth/demo-login`, both 200 — while
# every request carrying a token in that header reached the server with nothing: `/api/auth/me`
# answered as a guest to a browser that was holding a token, `/api/billing/account` answered
# 401, and minting a new token changed neither, because the new one went missing the same way.
# `tools/auth_stripping_proxy.py` reproduces it, and is how the fix was checked rather than
# assumed.


def test_the_alternate_header_authenticates_through_a_proxy_that_strips_authorization(signed_in):
    """The reproduction, kept as a test so it cannot come back.

    `Authorization` is dropped between the browser and the application, which is exactly what
    the preview does; the token arrives in the plain header instead and the account is found.
    """
    import importlib.util
    import threading

    spec = importlib.util.spec_from_file_location(
        "auth_stripping_proxy", ROOT / "tools" / "auth_stripping_proxy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.Proxy)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with TestClient(app) as client:
            token = client.post("/api/auth/demo-login",
                                json={"role": "user"}).json()["access_token"]

        import urllib.request as request
        def ask(headers):
            req = request.Request(f"http://127.0.0.1:{port}/api/auth/me", headers=headers)
            with request.urlopen(req) as answer:
                return json.loads(answer.read())

        # The control: the header the proxy eats, alone.
        assert ask({"Authorization": f"Bearer {token}"})["role"] == "guest"
        # The fix: the same token, in a header nothing has a reason to consume.
        assert ask({"X-Eidomira-Token": token})["email"] == "demo@eidomira.test"
    finally:
        server.shutdown()
        server.server_close()


def test_the_client_says_it_both_ways():
    """One credential, two headers, no fallback logic: whichever the deployment preserves is
    enough. A client that sends only `Authorization` is at the mercy of whatever is in front
    of it."""
    for name in ("app.js", "admin.js"):
        script = (ROOT / "static" / name).read_text(encoding="utf-8")
        assert "X-Eidomira-Token" in script, f"{name} has only one way to present the token"
        assert "Authorization" in script, f"{name} dropped the standard header"
    api = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "function sendToken(headers)" in api
    assert api.count("sendToken(") >= 4, "a call site still sets only one header"


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
    assert "a credential arrived via authorization and was refused" in line


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
    assert line and "via authorization and was accepted" in line


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
