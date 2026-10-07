"""How a page reaches a browser — two failures found by a user who could not find the login.

`/app` is where every "Open Studio" link on the marketing page lands, and a signed-out
browser was answered with a bare 401 holding no way forward. The login card is on `/`, and
nothing on the 401 said so, so the honest reading of that page was "there is no way in".
`static/app.js` had *always* meant to send people to `/?signin=1` for exactly this reason —
the line was there, and it never ran, because the server refused before the script was
served. The landing page, for its part, ignored the parameter. Both halves are pinned here.

The second failure is quieter: no page carried a cache directive, so a browser was entitled
to reuse what it already had instead of asking. A fix that is live on the server can stay
invisible in a tab that was open across the deploy, which is indistinguishable from a fix
that did not work.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent

# What a browser sends. The distinction matters: it is how the server tells a person looking
# for a page from a client calling an API, and the two want different answers.
BROWSER = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


def test_a_signed_out_browser_is_sent_to_the_login_card(isolated_db):
    with TestClient(app) as client:
        response = client.get("/app", headers=BROWSER, follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/?signin=1"


def test_an_api_client_still_gets_the_401_it_always_got(isolated_db):
    """The contract the redirect must not change: no HTML asked for, no HTML assumed."""
    with TestClient(app) as client:
        response = client.get("/app")

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_a_signed_in_browser_still_gets_the_workspace(isolated_db):
    """Registering returns no session — the address has to be verified first — so the sign-in
    is the step that produces the token this checks."""
    with TestClient(app) as client:
        email, password = "someone@example.com", "correct-horse-battery"
        assert client.post("/api/auth/register", json={"email": email, "password": password}).status_code == 200
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200
        token = login.json()["access_token"]
        response = client.get("/app", headers={**BROWSER, "authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert "authModal" in response.text


def test_the_login_card_does_not_take_over_a_signed_in_visitor():
    """A source check, because this suite has no DOM — the behaviour itself was confirmed by
    running the page's own script in one. It pins the two clauses that make the redirect safe
    to follow blindly: open the card only with no session, and drop the flag afterwards so a
    reload does not open it again. The unguarded version of this line predates the redirect
    and did exactly that to signed-in visitors."""
    script = (ROOT / "static" / "landing.js").read_text(encoding="utf-8")
    at = script.index('get("signin")')
    clause = script[at:at + 260]
    assert '!localStorage.getItem("eidomira_access_token")' in clause, \
        "the flag opens the card for a visitor who already has a session"
    assert "replaceState" in clause, "the flag stays in the URL and reopens the card on reload"


def test_pages_and_scripts_are_revalidated_rather_than_reused(isolated_db):
    """`no-cache`, not `no-store`: the ETag still answers most of these with a 304, so the
    cost is a round trip. What it removes is a browser's licence to invent a freshness
    lifetime for a page whose whole job is to show what the server currently offers."""
    with TestClient(app) as client:
        for path in ("/", "/static/app.html", "/static/app.js", "/static/landing.js", "/static/tokens.css"):
            response = client.get(path, headers=BROWSER, follow_redirects=False)
            assert response.headers.get("cache-control") == "no-cache", path


def test_the_api_is_still_never_stored(isolated_db):
    with TestClient(app) as client:
        assert client.get("/api/health").headers["cache-control"] == "no-store"
