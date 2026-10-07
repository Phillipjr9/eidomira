"""The owner console, and the role that opens it.

Four failures are worth guarding here, and three of them are the same failure — a
privilege that is decided somewhere other than the database:

* a role carried in the token, which would let a demoted administrator keep working until
  their token expired, and would make a leaked token a permanent grant;
* a role accepted from the registration form, which is the shortest path from "anyone can
  register" to "anyone is an administrator";
* a console that leaks the things it sits on top of — a password hash in a JSON payload is
  a password hash in a browser cache, a log line and a screenshot;
* an installation with no administrator at all, because the only one was demoted by a
  command that looked like it worked.

The last test in the file is about the page itself: it renders email addresses that people
chose, so it builds nodes instead of markup, and that rule is absolute enough to check.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import database
from app.main import app
from app.security import register as register_account
from tools.grant_admin import main as grant_admin

PASSWORD = "a-sufficiently-long-passphrase"
ROOT = Path(__file__).resolve().parent.parent
ADMIN_JS = (ROOT / "static" / "admin.js").read_text(encoding="utf-8")
ADMIN_HTML = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

#: Keys that must never appear anywhere in the console payload. Checked by name rather
#: than by value, because a secret that is currently empty would pass a value check and
#: start leaking the moment it was configured.
FORBIDDEN_KEYS = {
    "password_hash", "token_hash", "auth_secret", "paystack_secret_key", "smtp_password",
    "livekit_api_secret", "turn_secret", "provider_customer_id", "provider_subscription_id",
}


@pytest.fixture(autouse=True)
def _always_isolated(isolated_db):
    """Autouse on purpose, and the reason is a bug this file shipped with.

    Only the HTTP tests asked for `isolated_db`, and those were safe. The command-line
    tests did not — so they promoted accounts in the real `data/eidomira.db`, and left
    three of them as administrators on the machine. A fixture that has to be remembered
    is a fixture that will be forgotten, so this one cannot be.
    """
    yield


@pytest.fixture
def client(isolated_db):
    with TestClient(app) as test_client:
        yield test_client


def sign_up(client, email: str, password: str = PASSWORD, **extra):
    return client.post("/api/auth/register",
                       json={"email": email, "password": password, **extra})


def sign_in(client, email: str, password: str = PASSWORD):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def account(email: str) -> dict:
    return register_account(email, PASSWORD)


def run(capsys, argv: list[str]) -> tuple[int, str, str]:
    code = grant_admin(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def promote(capsys, email: str) -> None:
    code, _, _ = run(capsys, [email])
    assert code == 0, f"could not promote {email}"


def all_keys(node, prefix: str = "") -> set[str]:
    """Every key name in a JSON structure, at any depth, lists included."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            found |= all_keys(value, f"{prefix}.{key}")
    elif isinstance(node, list):
        for item in node:
            found |= all_keys(item, prefix)
    return found


# ── who gets in ───────────────────────────────────────────────────────────────

def test_anonymous_callers_are_turned_away(client):
    """The console must not be reachable by navigating to it, even though this build runs
    with accounts optional for the studio itself."""
    assert client.get("/admin").status_code == 401
    assert client.get("/api/admin/overview").status_code == 401


def test_an_ordinary_account_is_not_an_administrator(client):
    sign_up(client, "member@example.com")
    sign_in(client, "member@example.com")

    assert client.get("/admin").status_code == 403
    assert client.get("/api/admin/overview").status_code == 403


def test_promoting_an_account_opens_the_console(client, capsys):
    account("owner@example.com")
    code, out, _ = run(capsys, ["owner@example.com"])
    assert code == 0 and "promoted" in out

    sign_in(client, "owner@example.com")
    page = client.get("/admin")
    assert page.status_code == 200
    assert "owner console" in page.text
    # Never framed: the page shows account addresses and payment history.
    assert page.headers["x-frame-options"] == "SAMEORIGIN"

    payload = client.get("/api/admin/overview").json()
    assert {"generated_at", "accounts", "payments", "reports", "system", "live"} <= set(payload)
    assert payload["accounts"]["counts"]["admins"] == 1
    assert payload["accounts"]["accounts"][0]["email"] == "owner@example.com"
    assert payload["accounts"]["truncated"] is False


def test_a_disabled_administrator_loses_the_console(client, capsys):
    record = account("gone@example.com")
    promote(capsys, "gone@example.com")
    sign_in(client, "gone@example.com")
    assert client.get("/api/admin/overview").status_code == 200

    database.execute("UPDATE users SET disabled=1 WHERE id=?", (record["id"],))

    # Same still-unexpired cookie: disabling an account has to be enough on its own.
    assert client.get("/api/admin/overview").status_code == 401


# ── the role lives in the database, and nowhere else ──────────────────────────

def test_registration_cannot_choose_a_role(client):
    """The form is the one place an attacker already has a foothold, so the field is not
    read — sending it changes nothing rather than being rejected, which would be a hint."""
    response = sign_up(client, "sneaky@example.com", role="admin")
    assert response.status_code == 200

    stored = database.one("SELECT role FROM users WHERE email=?", ("sneaky@example.com",))
    assert stored["role"] == "user"
    assert database.one("SELECT COUNT(*) AS n FROM users WHERE role='admin'")["n"] == 0


def test_a_token_cannot_grant_what_the_database_does_not(client):
    """Signed with the real key, for a real account, claiming administrator. It still must
    not open the console, because the claim is never the input to the decision."""
    record = account("claimant@example.com")
    now = int(time.time())
    forged = jwt.encode(
        {"sub": record["id"], "email": record["email"], "role": "admin",
         "iat": now, "nbf": now - 2, "exp": now + 600, "typ": "access"},
        settings.auth_secret, algorithm="HS256",
    )

    response = client.get("/api/admin/overview",
                          headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 403


def test_demotion_applies_to_a_token_that_is_still_valid(client, capsys):
    account("temp@example.com")
    account("steady@example.com")
    promote(capsys, "temp@example.com")
    promote(capsys, "steady@example.com")

    sign_in(client, "temp@example.com")
    assert client.get("/api/admin/overview").status_code == 200

    code, out, _ = run(capsys, ["--demote", "temp@example.com"])
    assert code == 0 and "demoted" in out

    # No re-login, no waiting for expiry: the next request is already an ordinary one.
    assert client.get("/api/admin/overview").status_code == 403
    assert client.get("/admin").status_code == 403


def test_the_last_administrator_cannot_be_demoted(capsys):
    account("only@example.com")
    promote(capsys, "only@example.com")

    code, _, err = run(capsys, ["--demote", "only@example.com"])

    assert code == 1
    assert "last administrator" in err
    stored = database.one("SELECT role FROM users WHERE email=?", ("only@example.com",))
    assert stored["role"] == "admin"


# ── the tool ──────────────────────────────────────────────────────────────────

def test_the_tool_refuses_to_invent_an_account(capsys):
    """An account made from a shell prompt would have a password that travelled through a
    terminal, a history file or a chat log. Registration happens on the site."""
    code, _, err = run(capsys, ["ghost@example.com"])

    assert code == 1
    assert "register through the site" in err
    assert database.one("SELECT id FROM users WHERE email=?", ("ghost@example.com",)) is None


def test_listing_reports_roles_without_changing_them(capsys):
    account("a@example.com")
    account("b@example.com")
    promote(capsys, "a@example.com")

    code, out, _ = run(capsys, ["--list"])

    assert code == 0
    assert "a@example.com" in out and "administrator" in out
    assert "b@example.com" in out
    assert "1 administrator(s)" in out
    assert database.one("SELECT role FROM users WHERE email=?", ("b@example.com",))["role"] == "user"


def test_promoting_twice_is_not_an_error(capsys):
    account("again@example.com")
    promote(capsys, "again@example.com")
    code, out, _ = run(capsys, ["again@example.com"])
    assert code == 0 and "already an administrator" in out


# ── what the console is allowed to contain ────────────────────────────────────

def test_the_payload_carries_no_secret(client, capsys):
    account("owner@example.com")
    promote(capsys, "owner@example.com")
    sign_in(client, "owner@example.com")
    payload = client.get("/api/admin/overview").json()

    assert FORBIDDEN_KEYS.isdisjoint(all_keys(payload))

    blob = json.dumps(payload)
    assert settings.auth_secret not in blob
    for name in ("paystack_secret_key", "smtp_password", "livekit_api_secret", "turn_secret"):
        value = getattr(settings, name, "")
        if value:
            assert value not in blob, f"{name} reached the console payload"


def test_the_swap_verdict_matches_its_evidence(client, capsys):
    """can_swap is computed from the filesystem and the installed runtimes. If it ever
    disagreed with its own blocker list, the console would be reassuring an owner about a
    product that cannot run."""
    account("owner@example.com")
    promote(capsys, "owner@example.com")
    sign_in(client, "owner@example.com")
    system = client.get("/api/admin/overview").json()["system"]

    assert system["can_swap"] is (not system["blockers"])
    assert set(system["runtimes"]) == {"onnxruntime", "insightface", "torch"}
    assert isinstance(system["security"]["auth_secret_is_repository_default"], bool)
    assert system["engine"]["name"] == app.state.__dict__.get("engine_name", system["engine"]["name"])

    if not system["weights"]["configured_swap_present"]:
        assert any("swap weights" in reason for reason in system["blockers"]), (
            "a missing swap model must be named in the reasons, not left to the reader"
        )


# ── the page ──────────────────────────────────────────────────────────────────

def test_the_console_defines_every_id_its_script_reads():
    defined = set(re.findall(r'\bid="([^"]+)"', ADMIN_HTML))
    used = set(re.findall(r'\$\("([^"]+)"\)', ADMIN_JS))
    used |= set(re.findall(r'(?:fill|table)\("([^"]+)"', ADMIN_JS))

    assert used, "the console script no longer references the page — this test is worthless"
    assert used - defined == set(), f"the script reaches for ids that do not exist: {sorted(used - defined)}"


def test_the_console_builds_nodes_instead_of_markup():
    """Every value on this page comes from the database, and some of it is text a user
    chose. Interpolating an address into markup would make "which account is this" a
    stored cross-site scripting bug on the most privileged page in the product."""
    assert not re.search(r"\.innerHTML\s*=", ADMIN_JS)
    assert not re.search(r"\.outerHTML\s*=", ADMIN_JS)
    assert not re.search(r"insertAdjacentHTML|document\.write", ADMIN_JS)
