"""One-click demo sign-in, and the guarantee that it does not exist unless asked for.

The risk with this feature is not that it is inconvenient. It is that it is convenient, and
that it is the kind of thing somebody switches on to show a colleague, and then leaves on.
The failure is silent in the worst direction: the site works, the demo works, and whoever
reads the login page can be the administrator.

So most of these tests are about absence. With the flag unset the route does not exist, no
account is created at boot, the pages ship the demo row hidden, and neither page contains an
address or a password to find. Then, with the flag on, that the thing works — and still
publishes nothing: the button asks the server for a session rather than filling in a form,
and the passwords are random strings nobody is ever shown.

The last guard is the one that matters on a real deployment: turning the flag on over https
does nothing at all, and says why.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import demo
from app.billing import account as billing_account
from app.config import settings
from app.database import database
from app.main import app
from app.security import authenticate

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
DEMO_EMAILS = list(demo.ACCOUNTS.values())

#: What an owner might set as STUDIO_DEMO_PASSWORD: long enough for the ordinary rules, and
#: obvious about being a demo. The point of it is to be published, so the tests treat it the
#: way the card does.
CONFIGURED_PASSWORD = "demo-account-2026"


@pytest.fixture(autouse=True)
def _always_isolated(isolated_db):
    yield


@pytest.fixture
def demo_on(monkeypatch, isolated_db):
    """The flag on, a local URL, and the accounts seeded — the state a preview runs in."""
    monkeypatch.setattr(settings, "demo_login", True)
    monkeypatch.setattr(settings, "public_url", "http://127.0.0.1:8000")
    demo.ensure_accounts()
    return demo


@pytest.fixture
def demo_on_with_password(monkeypatch, isolated_db):
    """The flag on, a local URL, and a password the owner has chosen to publish."""
    monkeypatch.setattr(settings, "demo_login", True)
    monkeypatch.setattr(settings, "public_url", "http://127.0.0.1:8000")
    monkeypatch.setattr(settings, "demo_password", CONFIGURED_PASSWORD)
    demo.ensure_accounts()
    return demo


def page_text(*names: str) -> str:
    return "\n".join((STATIC / name).read_text(encoding="utf-8") for name in names)


def users() -> list[dict]:
    with database.lock, database.connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT id,email,role,disabled,email_verified_at FROM users ORDER BY email").fetchall()]


# ── absence is the default ────────────────────────────────────────────────────

def test_the_route_does_not_exist_until_somebody_enables_it(isolated_db):
    """404 rather than 403: an installation without demo sign-in should not advertise that
    it has one switched off."""
    with TestClient(app) as client:
        assert client.post("/api/auth/demo-login", json={"role": "user"}).status_code == 404
        methods = client.get("/api/auth/methods").json()
        assert methods["demo_login"] is False
        assert methods["demo_roles"] == []
        assert methods["demo_accounts"] == [], "a disabled demo must not name its accounts"


def client_get_method_not_allowed() -> int:
    with TestClient(app) as client:
        return client.get("/api/auth/demo-login").status_code


def test_a_disabled_installation_hides_the_route_from_every_role(isolated_db):
    """The 404 is only a 404 if it is the *first* thing checked. Validating the role first
    answered 400 to a body naming a role that does not exist, which tells a probe the route
    is present and merely switched off — the inference the 404 exists to deny."""
    with TestClient(app) as client:
        valid = client.post("/api/auth/demo-login", json={"role": "user"})
        unknown = client.post("/api/auth/demo-login", json={"role": "root"})
        empty = client.post("/api/auth/demo-login", json={})
        malformed = client.post("/api/auth/demo-login", content=b"not json")

    assert valid.status_code == unknown.status_code == empty.status_code == 404
    assert unknown.json() == valid.json()
    assert empty.json() == valid.json()

    # The malformed body is the edge of what this can promise, and it is the framework's
    # edge, not the route's: FastAPI validates the body before the handler runs, so it
    # answers 422. That is not a hole, because whether the *path* exists was never the
    # secret — a GET has always answered 405 — and the 404 here only refuses to confirm
    # the feature is present and switched off. Pinned so the limit is visible rather than
    # discovered later.
    assert malformed.status_code == 422
    assert client_get_method_not_allowed() == 405


def test_booting_with_the_flag_off_creates_nothing(isolated_db):
    """The bootstrap is the dangerous half: an account that exists is an account that can be
    signed into, whatever the pages say."""
    with TestClient(app) as client:
        client.get("/api/auth/methods")

    assert users() == []


def test_the_login_cards_ship_the_demo_row_hidden():
    """Hidden in the markup, not only by script: with JavaScript broken or blocked, an
    un-hidden row would be two buttons that lead nowhere, and the point is that nothing
    about the demo is visible until the server says it exists."""
    for name in ("app.html", "index.html"):
        markup = (STATIC / name).read_text(encoding="utf-8")
        row = re.search(r'<div class="demoRow" id="demoRow"[^>]*>', markup)
        assert row, f"{name} has no demo row"
        assert "hidden" in row.group(0), f"{name} ships the demo row visible"


def test_the_card_learns_the_addresses_from_the_server(demo_on):
    """The card names the account behind each button, and it can only do that while the server
    says the demo exists: the addresses are not written into the page, they arrive from the
    API. So a page can never advertise a demo account that is not there, and turning the flag
    off empties the row without anybody editing markup."""
    with TestClient(app) as client:
        methods = client.get("/api/auth/methods").json()

    assert [a["role"] for a in methods["demo_accounts"]] == ["user", "admin"]
    assert {a["email"] for a in methods["demo_accounts"]} == set(DEMO_EMAILS)
    for name in ("app.html", "index.html"):
        markup = (STATIC / name).read_text(encoding="utf-8")
        # One fill target per button, plus the script that fills them: if the markup loses
        # `data-email` the label would silently keep showing nothing.
        assert markup.count("data-email") == 2, f"{name} has no place to show the address"
    for name in ("app.js", "landing.js"):
        script = (STATIC / name).read_text(encoding="utf-8")
        assert "demo_accounts" in script, f"{name} never reads the addresses"


def test_no_page_carries_a_demo_credential():
    """No page *stores* a credential, and that is still the rule.

    Both the addresses and the password reach the card at run time, from an API that only
    offers them while the demo is on, so turning the flag off empties the card. If a future
    change writes either into the markup or into a script, this fails before a page does."""
    pages = page_text("app.html", "index.html", "app.js", "landing.js")
    for email in DEMO_EMAILS:
        assert email not in pages, f"{email} is written into a page"
    assert "eidomira.test" not in pages
    # Shape, not intent: an identifier ending in `password`, then `:` or `=`, then a quoted
    # value with no spaces and at least six characters — what an assigned credential looks
    # like, including `DEMO_PASSWORD = ...`, which a bare word-boundary check misses because
    # `_` is a word character. Prose that merely mentions the word is not a credential, and
    # requiring a single token is what separates the two. A committed password with a space in
    # it would slip past; the exact-value checks above cover the values actually in use.
    assert not re.search(r"""(?i)\w*password\s*[:=]\s*["'][^"'\s]{6,}["']""", pages), \
        "a password literal appears in a page"


# ── switching it on ───────────────────────────────────────────────────────────

def test_enabling_it_creates_one_account_per_role(demo_on):
    rows = users()
    assert [row["email"] for row in rows] == sorted(DEMO_EMAILS)

    by_email = {row["email"]: row for row in rows}
    assert by_email[demo.ACCOUNTS["user"]]["role"] == "user"
    assert by_email[demo.ACCOUNTS["admin"]]["role"] == "admin"
    for row in rows:
        assert row["email_verified_at"], "an unverified demo account cannot buy anything"
        assert not row["disabled"]

    # The ordinary verified-email trial, so the demo shows the real thing.
    for row in rows:
        plan = billing_account(row["email"] and _user_id(row["email"]))
        assert plan["wallet"]["total"] == 100
        assert plan["subscription"]["plan"] == "trial"


def _user_id(email: str) -> str:
    return database.one("SELECT id FROM users WHERE email=?", (email,))["id"]


def test_seeding_twice_does_not_stack_credits(demo_on):
    """Boot runs this on every restart, so it has to be safe to run on every restart."""
    before = {email: billing_account(_user_id(email))["wallet"]["total"] for email in DEMO_EMAILS}

    assert demo.ensure_accounts() == []

    after = {email: billing_account(_user_id(email))["wallet"]["total"] for email in DEMO_EMAILS}
    assert after == before
    assert len(users()) == len(DEMO_EMAILS)


def test_one_click_signs_in_without_a_password(demo_on):
    with TestClient(app) as client:
        response = client.post("/api/auth/demo-login", json={"role": "user"})

        assert response.status_code == 200
        body = response.json()
        assert body["user"]["email"] == demo.ACCOUNTS["user"]
        assert body["user"]["role"] == "user"
        assert body["access_token"]

        # The cookie is sent for the API paths that accept one — and the token in the body is
        # what a browser actually carries, since a navigation cannot send a header.
        assert client.get("/api/auth/me").json()["role"] == "user"
        assert client.get("/app").status_code == 200
        # The console refuses this account's *data*; the page itself is a shell served to
        # anyone, and the console's own script is what turns an ordinary account away.
        assert client.get("/api/admin/overview").status_code == 403
        assert client.get("/admin").status_code == 200


def test_the_admin_button_reaches_the_owner_console(demo_on):
    with TestClient(app) as client:
        response = client.post("/api/auth/demo-login", json={"role": "admin"})

        assert response.status_code == 200
        assert response.json()["user"]["role"] == "admin"
        assert client.get("/admin").status_code == 200
        assert client.get("/api/admin/overview").status_code == 200


def test_the_addresses_are_marked_as_demo_accounts_in_the_console(demo_on):
    """So an owner reading the console does not count the furniture as customers."""
    with TestClient(app) as client:
        client.post("/api/auth/demo-login", json={"role": "admin"})
        overview = client.get("/api/admin/overview").json()

    assert overview["accounts"]["counts"]["demo"] == len(DEMO_EMAILS)
    assert all(row["demo"] for row in overview["accounts"]["accounts"])
    assert overview["system"]["security"]["demo_login"] is True


def test_an_unknown_role_is_refused(demo_on):
    with TestClient(app) as client:
        assert client.post("/api/auth/demo-login", json={"role": "root"}).status_code == 400


def test_an_omitted_role_gets_the_customer_and_never_the_administrator(demo_on):
    """`role` defaults to the customer. A default that picked the administrator would mean a
    request nobody wrote deliberately could open the owner console."""
    with TestClient(app) as client:
        response = client.post("/api/auth/demo-login", json={})

    assert response.status_code == 200
    assert response.json()["user"]["email"] == demo.ACCOUNTS["user"]
    assert response.json()["user"]["role"] == "user"


# ── a session that outlives the database ──────────────────────────────────────
#
# Measured in this repository's own preview, repeatedly: the database does not survive a
# restart of the host, the demo accounts were rebuilt with fresh random ids, and every token
# minted before the rebuild named a row that no longer existed. The server answered 401, the
# page signed the visitor out, and signing in again only bought time until the next rebuild.

def test_the_demo_accounts_are_created_with_fixed_ids(demo_on):
    rows = {row["email"]: row["id"] for row in users()}
    assert rows[demo.ACCOUNTS["user"]] == demo.DEMO_IDS["user"]
    assert rows[demo.ACCOUNTS["admin"]] == demo.DEMO_IDS["admin"]
    # Opaque, and the right shape for the column: the id is a string, not a number.
    for user_id in demo.DEMO_IDS.values():
        assert isinstance(user_id, str) and user_id


def test_signing_in_survives_the_database_being_rebuilt(demo_on):
    """The whole point. A token minted before the accounts are recreated must still work
    afterwards, which is only true if the recreated account has the same id and the signing
    key has not changed."""
    with TestClient(app) as client:
        response = client.post("/api/auth/demo-login", json={"role": "admin"})
        assert response.status_code == 200
        token = response.json()["access_token"]
        assert client.get("/api/auth/me",
                          headers={"authorization": f"Bearer {token}"}).json()["email"] \
            == demo.ACCOUNTS["admin"]

        # A rebuild: every account row is gone, exactly as a wiped volume leaves it, and the
        # bootstrap runs again on the next start.
        database.execute("DELETE FROM subscriptions")
        database.execute("DELETE FROM credit_wallets")
        database.execute("DELETE FROM credit_ledger")
        database.execute("DELETE FROM users")
        assert users() == []
        created = demo.ensure_accounts()
        assert sorted(created) == sorted(DEMO_EMAILS)

        # The same token, against the rebuilt account.
        again = client.get("/api/auth/me", headers={"authorization": f"Bearer {token}"})
        assert again.status_code == 200, "the session did not survive the rebuild"
        assert again.json()["email"] == demo.ACCOUNTS["admin"]
        assert again.json()["role"] == "admin"


def test_an_account_that_already_exists_keeps_its_id(demo_on):
    """Changing an existing id would strand every wallet, subscription and ledger row that
    references it, so the bootstrap never moves one."""
    before = {row["email"]: row["id"] for row in users()}
    database.execute("UPDATE users SET id='an-explicitly-different-id' WHERE email=?",
                     (demo.ACCOUNTS["user"],))

    assert demo.ensure_accounts() == []

    after = {row["email"]: row["id"] for row in users()}
    assert after[demo.ACCOUNTS["user"]] == "an-explicitly-different-id"
    assert after[demo.ACCOUNTS["admin"]] == before[demo.ACCOUNTS["admin"]]


# ── a browser that lost its storage is not a signed-out visitor ─────────────────
#
# The preview is served from `https://<port>-<sandbox>.e2b.app`, and the sandbox identifier
# changed between two sessions of work on this feature: `iifsolp0din8orm9w5hce` became
# `ipv1b4xm4krvvtj8q6qdo`. localStorage is per-origin, so a changed host is a changed site and
# the token stored under the old one is invisible. A rebuilt database does the same thing. Both
# present as "it logged me out again", and both are recoverable while the demo is on — a flag
# that already means "anyone who can reach this can sign in" and already refuses to run over
# https.

def test_the_studio_can_put_a_lost_demo_session_back(demo_on):
    """A request with no token at all is what a changed origin looks like to the server. The
    studio asks for a session and carries on, instead of treating it as the end."""
    with TestClient(app) as client:
        minted = client.post("/api/auth/demo-login", json={"role": "user"})
        assert minted.status_code == 200
        token = minted.json()["access_token"]

        assert client.get("/api/auth/me",
                          headers={"authorization": f"Bearer {token}"}).json()["email"] \
            == demo.ACCOUNTS["user"]


def test_a_lost_session_is_restored_to_the_same_role(demo_on):
    with TestClient(app) as client:
        restored = client.post("/api/auth/demo-login", json={"role": "admin"})

    assert restored.status_code == 200
    assert restored.json()["user"]["role"] == "admin"
    assert restored.json()["user"]["email"] == demo.ACCOUNTS["admin"]


def test_the_client_restores_before_it_signs_anybody_out():
    """Pinned in the source, since this suite has no DOM. The order is the whole fix."""
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "async function restoreDemoSession()" in script
    assert "if(restoreAttempted)return false" in script, "a failed restore could loop"
    assert "DEMO_ROLE_KEY" in script, "the restored session is not the same role"

    # Nothing may decide the session is over before the restore is attempted. A redirect on the
    # line that reads the token is how this failed a third time: the page navigated to the
    # sign-in card while the restore was still in flight, so the browser followed the
    # navigation and the session was put back on a page nobody was looking at any more. The
    # redirect now exists only inside that one decision, which fires once and is never reached
    # at load time.
    head = script[:script.index("async function apiFetch")]
    assert "if(!accessToken)location" not in script, "a statement redirects before the restore"
    assert head.count("location.replace") == 1, "a redirect is reachable before the decision"
    assert "function endSession(" in head and "if(sessionEnded)return" in head

    boot = script[script.index("async function loadAccount()"):]
    assert "if(!accessToken&&!(await restoreDemoSession()))" in boot, \
        "an empty token store ends the session without trying to restore"

    api = script[script.index("async function apiFetch"):script.index("async function sessionUser()")]
    assert api.index("restoreDemoSession") < api.index("endSession("), \
        "a rejected token signs the visitor out before restore is attempted"


def test_the_demo_buttons_use_the_password_path():
    """Asked for directly: the demo uses the password too. The private minting endpoint stays
    as a fallback so a button cannot dead-end on a deployment where sign-in itself is broken."""
    for name, quote in (("app.js", "'"), ("landing.js", '"')):
        script = (STATIC / name).read_text(encoding="utf-8")
        assert f"{quote}/api/auth/login{quote}" in script, f"{name} does not use the password path"
        assert "demo_password" in script, f"{name} does not know the password"
        assert "/api/auth/demo-login" in script, f"{name} lost its fallback"


# ── the optional password an owner may choose to publish ──────────────────────

def test_a_configured_password_signs_in_on_the_ordinary_form(demo_on_with_password):
    """The point of the setting: the accounts can be typed into the normal sign-in, not only
    opened by the button — so the demo also demonstrates the real flow."""
    for email in DEMO_EMAILS:
        user = authenticate(email, CONFIGURED_PASSWORD)
        assert user and user["email"] == email

    with TestClient(app) as client:
        response = client.post("/api/auth/login",
                               json={"email": demo.ACCOUNTS["admin"], "password": CONFIGURED_PASSWORD})

    assert response.status_code == 200
    assert response.json()["user"]["role"] == "admin"


def test_the_configured_password_is_the_only_one_that_works(demo_on_with_password):
    for guess in ("demo", "admin", "demo-account", CONFIGURED_PASSWORD + "x", ""):
        assert authenticate(DEMO_EMAILS[0], guess) is None, f"accepted {guess!r}"


def test_the_card_is_offered_the_password_when_one_is_configured(demo_on_with_password):
    with TestClient(app) as client:
        methods = client.get("/api/auth/methods").json()

    assert methods["demo_password"] == CONFIGURED_PASSWORD
    for name in ("app.js", "landing.js"):
        assert "demo_password" in (STATIC / name).read_text(encoding="utf-8"), \
            f"{name} never reads the configured password"
    for name in ("app.html", "index.html"):
        markup = (STATIC / name).read_text(encoding="utf-8")
        assert 'id="demoHint"' in markup, f"{name} has nowhere to print it"


def test_the_card_is_always_told_the_password_while_the_demo_is_on(demo_on):
    with TestClient(app) as client:
        methods = client.get("/api/auth/methods").json()

    assert methods["demo_password"] == demo.DEFAULT_PASSWORD


def test_the_password_is_withheld_while_the_demo_is_off(monkeypatch, isolated_db):
    """Set but not switched on is not a reason to hand a credential to a public API."""
    monkeypatch.setattr(settings, "demo_password", CONFIGURED_PASSWORD)

    with TestClient(app) as client:
        methods = client.get("/api/auth/methods").json()

    assert methods["demo_login"] is False
    assert methods["demo_password"] is None
    assert CONFIGURED_PASSWORD not in page_text("app.html", "index.html", "app.js", "landing.js")


def test_changing_the_password_updates_accounts_that_already_exist(demo_on, monkeypatch):
    """Restarting is the only step in changing it: the accounts are not recreated, and the
    credits somebody was looking at are not reset."""
    assert authenticate(DEMO_EMAILS[1], CONFIGURED_PASSWORD) is None
    before = {row["email"]: row for row in users()}

    monkeypatch.setattr(settings, "demo_password", CONFIGURED_PASSWORD)
    assert demo.ensure_accounts() == []

    assert authenticate(DEMO_EMAILS[1], CONFIGURED_PASSWORD)
    assert {row["email"]: row for row in users()} == before


def test_a_password_the_ordinary_rules_refuse_switches_the_demo_off(monkeypatch, isolated_db):
    """Short, so `register` would refuse it. Refused at the flag instead of crashing at boot,
    and loudly: the alternative is a startup traceback nobody reads until the page is broken."""
    monkeypatch.setattr(settings, "demo_login", True)
    monkeypatch.setattr(settings, "public_url", "http://127.0.0.1:8000")
    monkeypatch.setattr(settings, "demo_password", "short")

    assert demo.enabled() is False
    reason = demo.refusal_reason()
    assert reason and "10" in reason

    with TestClient(app) as client:
        assert client.post("/api/auth/demo-login", json={"role": "user"}).status_code == 404
        methods = client.get("/api/auth/methods").json()

    assert methods["demo_login"] is False
    assert methods["demo_password"] is None
    assert users() == []


def test_the_demo_passwords_are_not_guessable(demo_on):
    """The published password is the only thing that opens these accounts.

    They are meant to be signed into — that is what a demo account is — so the useful assertion
    is not that no password exists but that nothing else works: not the address, not the role,
    not the obvious guesses, and not a near miss on the real one.
    """
    published = demo.published_password()
    for email in DEMO_EMAILS:
        for guess in ("demo", "admin", "password", "demo1234", "eidomira-demo-2025",
                      published + "x", published[:-1], published.upper(), DEMO_EMAILS[0], ""):
            assert authenticate(email, guess) is None, f"{email} accepted {guess!r}"
        assert authenticate(email, published), "the published password must work"

    tree = ast.parse((ROOT / "app" / "demo.py").read_text(encoding="utf-8"))
    literals = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                literals[target.id] = node.value.value

    # Exactly one password lives in the source, it is the published default, and it is named
    # the way the docstring above it says. Anything else appearing here is a credential nobody
    # decided to publish, which is the mistake this check exists to catch.
    password_literals = {name: value for name, value in literals.items()
                         if "password" in name.lower() or "secret" in name.lower()
                         or "token" in name.lower()}
    assert set(password_literals) == {"DEFAULT_PASSWORD"}, password_literals
    assert password_literals["DEFAULT_PASSWORD"] == demo.DEFAULT_PASSWORD
    assert len(demo.DEFAULT_PASSWORD) >= 10


# ── the guard that matters on a real deployment ───────────────────────────────

def test_https_refuses_to_enable_demo_sign_in(monkeypatch, isolated_db):
    """The combination nobody means to ship: one-click administrator access on a public
    page. Warned about, this would still be on; so it is refused."""
    monkeypatch.setattr(settings, "demo_login", True)
    monkeypatch.setattr(settings, "public_url", "https://eidomira.example")

    assert demo.enabled() is False
    assert "https" in (demo.refusal_reason() or "")

    with TestClient(app) as client:
        assert client.post("/api/auth/demo-login", json={"role": "admin"}).status_code == 404
        assert client.get("/api/auth/methods").json()["demo_login"] is False

    assert users() == [], "the accounts were created even though the feature is refused"


def test_the_refusal_says_what_to_do_instead(monkeypatch, isolated_db):
    monkeypatch.setattr(settings, "demo_login", True)
    monkeypatch.setattr(settings, "public_url", "https://eidomira.example")
    reason = demo.refusal_reason()
    assert "real accounts" in reason or "local URL" in reason


def test_it_is_off_in_the_configuration_that_gets_deployed():
    """The shipped default, asserted rather than assumed, because the whole design rests on
    it: everything else here is a convenience, and this is the safety."""
    from app.config import Settings
    assert Settings().demo_login is False
