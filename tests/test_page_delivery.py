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

import hashlib
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app import pages
from app.main import app

ROOT = Path(__file__).resolve().parent.parent
static = ROOT / "static"

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


# ── the stamp that defeats a cache ────────────────────────────────────────────
#
# The login row was invisible for a day because of this class of failure: the page arrived
# (revalidated) and the script that makes it work did not (reused from the browser's own
# cache), so no request for it ever reached the server. A URL that changes when its content
# changes cannot be answered from a cache holding different content.

def live_markup(markup: str) -> str:
    """The page without its HTML comments.

    The result pane carries a commented-out example of the image that will go there, and its
    `src="/static/your-result.jpg"` is a live reference to a scanner — there is no such file,
    and stamping it would invent a URL. Same rule the landing-asset tests apply.
    """
    return re.sub(r"<!--.*?-->", "", markup, flags=re.S)


def test_served_pages_reference_their_assets_by_content_digest(isolated_db):
    """Every local asset reference is stamped with the real digest of the file, so the URL a
    page hands the browser cannot be answered with different bytes."""
    with TestClient(app) as client:
        for page in ("/", "/lab"):
            markup = live_markup(client.get(page).text)

            stamped = re.findall(r'"/static/([^"?]+)\?v=([0-9a-f]+)"', markup)
            assert stamped, f"{page} stamps nothing"
            for relative, digest in stamped:
                content = (static / relative).read_bytes()
                assert digest == hashlib.sha256(content).hexdigest()[:10], \
                    f"{page} stamps {relative} with the wrong digest"

            # Nothing that exists on disk is left unstamped, so nothing is left to a cache.
            unstamped = [ref for ref in re.findall(r'"/static/([^"?]+)"', markup)
                         if (static / ref).exists()]
            assert not unstamped, f"{page} leaves {unstamped} unstamped"

            # And the stamped URL is one the server actually serves.
            first = stamped[0]
            assert client.get(f"/static/{first[0]}?v={first[1]}").status_code == 200


def test_the_stamp_follows_the_file(tmp_path, monkeypatch):
    """Not a constant that happens to be right today: change the file and the URL follows."""
    store = tmp_path / "static"
    store.mkdir()
    monkeypatch.setattr(pages, "STATIC", store)
    asset = store / "thing.js"
    asset.write_text("one", encoding="utf-8")

    first = pages.stamp('<script src="/static/thing.js"></script>')
    assert f'v={hashlib.sha256(b"one").hexdigest()[:10]}' in first

    asset.write_text("two", encoding="utf-8")
    second = pages.stamp('<script src="/static/thing.js"></script>')
    assert second != first
    assert f'v={hashlib.sha256(b"two").hexdigest()[:10]}' in second


def test_a_reference_to_a_file_that_is_not_there_is_left_alone(tmp_path, monkeypatch):
    """Stamping a missing file would invent a URL for something that cannot be fetched. The
    404 it produces today is the useful signal."""
    monkeypatch.setattr(pages, "STATIC", tmp_path)
    reference = '<link rel="stylesheet" href="/static/absent.css">'
    assert pages.stamp(reference) == reference


def test_every_html_page_is_served_through_the_stamp():
    """A new HTML route added with FileResponse would silently skip the stamping, and that is
    exactly how this bug comes back. Cheap to check, and it fails loudly."""
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    direct = [line.strip() for line in source.splitlines()
              if "FileResponse(" in line and ".html" in line]
    assert not direct, f"these pages bypass the stamping: {direct}"

    for page in ("index.html", "app.html", "admin.html", "boost-lab.html"):
        assert re.search(rf"pages\.html\([^)]*{re.escape(page)}", source), \
            f"{page} is not served through pages.html"


def test_the_service_worker_precaches_only_what_exists():
    """`cache.addAll` rejects as a whole if one entry 404s, and the failure is swallowed at
    install — so a stale entry means no precache at all, discovered offline."""
    script = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
    shell = re.search(r"SHELL=\[(.*?)\];", script, re.S).group(1)
    paths = re.findall(r"'([^']+)'", shell)
    assert paths, "the precache list is empty"
    for path in paths:
        if path == "/":
            continue
        assert (ROOT / path.lstrip("/")).exists(), f"precached but missing: {path}"


def test_navigations_do_not_reuse_the_http_cache():
    """Assets are covered by their stamped URLs; the page itself is the entry point that
    carries the stamps, so it is the one request that must always go to the network."""
    script = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
    assert "cache:'reload'" in script, "the service worker may serve a stale page"


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
