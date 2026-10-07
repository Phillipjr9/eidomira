"""How a page reaches a browser, and what a page is allowed to decide.

Three failures live here, all found by a user who could not get in.

The first: `/app` is where every "Open Studio" link lands, and a signed-out browser was
answered with a bare 401 holding no way forward. The login card is on `/`; nothing said so.

The second: that was "fixed" by gating the page on the session cookie, which cannot work in
this product. The session lives in `localStorage` and travels as an `Authorization` header on
API calls, and a *navigation* cannot send a header — so the gate had only the cookie, and in
an embedded context the cookie never arrives. The preview proved it: `POST /api/auth/demo-login`
answered 200, the next `GET /app` arrived with no cookie, and the user was sent back to the
card in a loop. The page is now served to anyone and the *client* decides, because the client
is the only side holding the token. What makes that safe is the rule that always applied and
is pinned below: the shell is a static file, and the data behind it needs the session.

The third is quieter: no page carried a cache directive, so a browser was entitled to reuse
what it already had. A fix that is live on the server can stay invisible in a tab that was
open across the deploy, which is indistinguishable from a fix that did not work.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app import pages
from app.config import settings
from app.main import app

ROOT = Path(__file__).resolve().parent.parent
static = ROOT / "static"

# What a browser sends. The distinction matters: it is how the server tells a person looking
# for a page from a client calling an API, and the two want different answers.
BROWSER = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


def test_the_studio_shell_is_served_without_a_session(isolated_db):
    """The regression that mattered, in the shape it happened.

    A browser that had just signed in carried the session in `localStorage` and no cookie at
    all. A server-side gate can only see cookies, so it refused a user who was, by every
    measure this product uses, signed in. The page must load; `static/app.js` reads the token
    and redirects when there is not one — its first executable line, pinned below.
    """
    with TestClient(app) as client:
        assert client.get("/app", headers=BROWSER, follow_redirects=False).status_code == 200
        assert client.get("/app").status_code == 200, "an API client gets the shell too"
        assert client.get("/admin", headers=BROWSER).status_code == 200


def test_the_shell_holds_nothing_worth_hiding(isolated_db):
    """Why serving it to anyone is not a leak: it is the same static file the browser could
    always fetch at /static/app.html, it names no account, and it carries no session.

    A `type="password"` input is expected and is not a credential — it is the box somebody
    types into. What must not be here is a *filled-in* one, a token, or an address.
    """
    with TestClient(app) as client:
        shell = client.get("/app").text

    assert not re.search(r"eyJ[A-Za-z0-9_-]{10,}\.", shell), "the shell carries a token"
    assert "eidomira_access_token" not in shell, "the shell names the session key"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", shell), "the shell names an address"
    assert not re.search(r'(?i)<input[^>]+type="password"[^>]*value="[^"]+"', shell), \
        "a password is written into the shell"
    assert (static / "app.html").exists()


def test_the_data_behind_the_shell_needs_a_session(isolated_db):
    """The gate moved, it did not disappear. Every figure the studio shows comes from an API
    that still refuses an anonymous caller, and the console's from one that refuses everyone
    who is not an administrator."""
    with TestClient(app) as client:
        assert client.get("/api/billing/account").status_code == 401
        assert client.get("/api/admin/overview").status_code == 401

        email, password = "someone@example.com", "correct-horse-battery"
        assert client.post("/api/auth/register", json={"email": email, "password": password}).status_code == 200
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200
        token = login.json()["access_token"]

        assert client.get("/api/billing/account",
                          headers={"authorization": f"Bearer {token}"}).status_code == 200
        assert client.get("/api/admin/overview",
                          headers={"authorization": f"Bearer {token}"}).status_code == 403


def test_the_client_is_what_decides_who_sees_the_studio():
    """The other half of the move, and the half a server test cannot exercise: the studio's
    script redirects to the sign-in card when there is no token, and the console's script
    sends the token it holds rather than hoping for a cookie."""
    studio = (static / "app.js").read_text(encoding="utf-8")
    assert "location.replace('/?signin=1')" in studio, \
        "a signed-out visitor would see the studio shell and no way to sign in"
    assert "localStorage.getItem('eidomira_access_token')" in studio

    console = (static / "admin.js").read_text(encoding="utf-8")
    assert "localStorage.getItem(TOKEN_KEY)" in console
    assert 'headers.set("Authorization", "Bearer " + session)' in console
    # Every request the console makes goes through apiFetch; a bare fetch() there would be a
    # request without the session, which is the bug this replaced.
    assert 'fetch("/api/' not in console, "the console calls an API without its session"


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


# ── who is allowed to put the studio in a frame ───────────────────────────────
#
# The studio's page refuses to render inside another site's frame, because its buttons are
# worth stealing clicks for. A hosted preview is served inside a frame, so there the default
# refuses to render it at all — which is why an operator who embeds it says so explicitly.

def test_the_private_pages_refuse_to_be_framed_by_default(isolated_db):
    with TestClient(app) as client:
        for page in ("/app", "/admin"):
            headers = client.get(page, headers=BROWSER).headers
            assert "frame-ancestors 'self'" in headers["content-security-policy"], page
            assert headers["x-frame-options"] == "SAMEORIGIN", page


def test_the_marketing_page_is_framed_by_design(isolated_db):
    """Product embeds and hosted previews show it, so it stays embeddable — and it holds no
    session, which is what makes that safe."""
    with TestClient(app) as client:
        headers = client.get("/", headers=BROWSER).headers

    assert "frame-ancestors *" in headers["content-security-policy"]
    assert "x-frame-options" not in headers


def test_a_configured_ancestor_is_allowed_and_x_frame_options_stands_down(monkeypatch, isolated_db):
    """X-Frame-Options cannot express a list and would veto the allowed frame, so it is sent
    only where it agrees with the policy."""
    monkeypatch.setattr(settings, "embed_ancestors", "https://*.e2b.app, https://docs.example.com")

    with TestClient(app) as client:
        headers = client.get("/app", headers=BROWSER).headers
        marketing = client.get("/", headers=BROWSER).headers

    assert "frame-ancestors https://*.e2b.app https://docs.example.com" in headers["content-security-policy"]
    assert "x-frame-options" not in headers
    assert "frame-ancestors *" in marketing["content-security-policy"]


def test_an_empty_setting_is_the_default_and_an_empty_value_stays_closed(monkeypatch, isolated_db):
    """Whitespace is not a permission: `STUDIO_EMBED_ANCESTORS=" "` must not open the door."""
    monkeypatch.setattr(settings, "embed_ancestors", "   ")

    with TestClient(app) as client:
        headers = client.get("/app", headers=BROWSER).headers

    assert "frame-ancestors 'self'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "SAMEORIGIN"


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
