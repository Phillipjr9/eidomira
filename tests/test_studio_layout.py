"""The studio's shape: three sections, each thing in one place, nothing repeated.

The page this replaces was one long column. The balance was in the top bar and again in a
panel, the plan and its upgrade button were in an account strip and again in that panel, and
prices lived in three of those copies. Settings did not exist at all, and the packs were
hidden outright when Paystack was not configured — which is why a deployment with no payment
key had no payment page, and looked like a missing feature rather than a missing setting.

These are static checks on the markup and the script. They are cheap and they fail the moment
somebody adds a fourth copy, which is exactly when nobody notices.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
APP_HTML = (STATIC / "app.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
STUDIO_CSS = (STATIC / "studio.css").read_text(encoding="utf-8")


def view(name: str) -> str:
    """The markup of one section, so a check can say *where* something lives.

    Counts nesting rather than stopping at the first `</section>`: each view is a section that
    contains sections, and the naive slice returns only its first card — which is how this
    helper failed the first time it ran.
    """
    start = APP_HTML.index(f'id="view{name}"')
    body = APP_HTML.index(">", start) + 1
    depth = 1
    for match in re.finditer(r"</?section\b", APP_HTML[body:]):
        depth += -1 if match.group(0) == "</section" else 1
        if depth == 0:
            return APP_HTML[start:body + match.end()]
    raise AssertionError(f"section view{name} is never closed")


def ids(markup: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', markup))


# ── the sections themselves ───────────────────────────────────────────────────

def test_the_navigation_and_the_sections_agree():
    """A tab whose target does not exist is a dead link, and a section with no tab is
    reachable only by typing a hash."""
    views = set(re.findall(r'<section class="view" id="view(\w+)"', APP_HTML))
    tabs = set(re.findall(r'data-view="(\w+)"', APP_HTML))

    # The ids are capitalised (`viewBilling`) and the hrefs are not (`#billing`), so compare
    # them the way the router does: by lowercasing the id's tail.
    assert views == {name.capitalize() for name in tabs} == {"Studio", "Billing", "Settings"}

    # Billing and Settings ship hidden; the studio does not, because it is the default and a
    # visitor whose script failed should still see the workbench rather than a blank page.
    for name in ("Billing", "Settings"):
        assert re.search(rf'<section class="view" id="view{name}"[^>]*hidden', APP_HTML), name
    assert not re.search(r'<section class="view" id="viewStudio"[^>]*hidden', APP_HTML)


def test_two_sections_start_hidden_and_the_router_shows_one_at_a_time():
    assert APP_JS.count("VIEWS.forEach(v=>{$(viewId(v)).hidden=v!==wanted})") == 1
    assert 'addEventListener("hashchange"' in APP_JS or "addEventListener('hashchange'" in APP_JS
    assert "showView((location.hash" in APP_JS, "the URL does not decide the section"


# ── nothing is duplicated ─────────────────────────────────────────────────────

def test_the_balance_is_rendered_in_one_place():
    """The bar shows it as a link to Billing — a number and a pointer, not a second panel."""
    assert APP_HTML.count('id="creditBalance"') == 1
    assert "walletTotal" not in view("Studio"), "the studio carries a second balance"
    assert APP_HTML.count('id="walletTotal"') == 1
    assert "$('creditBalance').textContent=creditCount(total)" in APP_JS


def test_the_plan_is_rendered_in_one_place():
    assert APP_HTML.count('id="accountPlan"') == 1
    assert 'id="accountPlan"' in view("Billing"), "the plan belongs to Billing"
    assert "accountPlan" not in view("Settings"), "Settings repeats the plan"
    assert "planName" not in view("Studio") and "planPriceNgn" not in view("Studio")


def test_no_price_is_written_into_the_page_or_the_script():
    """Every figure the customer sees comes from /api/billing/account, which is the same
    payload that decides what Paystack is asked to charge. Typed-in prices are the bug that
    test_credits.py already had to fix once."""
    for markup, name in ((APP_HTML, "app.html"), (APP_JS, "app.js")):
        assert "\u20a6" not in markup.replace("\u20a6'+creditCount(plan.price_ngn)", ""), \
            f"{name} contains a typed naira amount"
        assert not re.search(r"\d[\d,]*\s*kobo", markup, re.I), f"{name} contains a typed amount"
    assert "billing.topups" in APP_JS, "the packs no longer come from the server"


def test_the_upgrade_and_pack_buttons_exist_once():
    for element in ("upgradeBtn", "annualBtn", "walletPacks"):
        assert APP_HTML.count(f'id="{element}"') == 1, element
        assert f'id="{element}"' in view("Billing"), f"{element} left the billing page"


# ── the payment page, which did not exist ─────────────────────────────────────

def test_the_packs_are_never_hidden_when_payments_are_off():
    """The regression that made the payment page invisible: `packs.hidden = true` whenever no
    Paystack key was configured, so the deployment with no key showed nothing at all. A
    disabled button that explains itself beats an absent one."""
    assert "packs.hidden" not in APP_JS, "the pack grid can still be hidden"
    assert "$('walletPacks')" in APP_JS
    assert 'id="paystackState"' in APP_HTML and 'id="paystackNote"' in APP_HTML
    assert "STUDIO_PAYSTACK_SECRET_KEY" in APP_JS, "nothing says how to switch payments on"


def test_each_pack_offers_a_way_to_pay_and_says_what_stops_it():
    assert "startCheckout(pack.product)" in APP_JS, "a pack has no checkout"
    assert "'Pay with Paystack'" in APP_JS
    assert "Verify your email first" in APP_JS, "the verified-email gate is gone"
    assert "authorization_url" in APP_JS, "checkout no longer redirects to the provider"


def test_the_packs_are_disabled_rather_than_missing_when_they_cannot_be_bought():
    assert "card.disabled=!buyable" in APP_JS
    assert "const buyable=configured&&verified" in APP_JS


# ── complete settings ────────────────────────────────────────────────────────

@pytest.mark.parametrize("card", ["Account", "Security", "Preferences", "Privacy and data",
                                  "Engine and diagnostics"])
def test_the_settings_page_has_each_card(card):
    assert f'aria-label="{card}"' in view("Settings"), card


def test_settings_covers_the_things_a_settings_page_owes_the_holder():
    """One id per promise: who you are signed in as, how to leave, how to change the password,
    what is stored on this device, and what the engine actually is."""
    settings = ids(view("Settings"))
    for element in ("accountEmail", "accountVerified", "accountRole", "resendBtn",
                    "logoutBtn", "sessionExpiry", "passwordForm", "pwCurrent", "pwNew",
                    "pwConfirm", "pwSubmit", "cameraDevice", "micDevice", "motionToggle",
                    "forgetBtn", "privacyTtl", "diagBackend", "diagProvider", "diagAccelerated",
                    "diagNote"):
        assert element in settings, f"settings has no {element}"


def test_the_password_form_posts_to_the_real_endpoint():
    assert "'/api/auth/password'" in APP_JS
    assert "current_password" in APP_JS and "new_password" in APP_JS
    assert "The two new passwords do not match." in APP_JS


def test_preferences_say_they_are_this_browser_only():
    """They are localStorage, not account settings, and the page must not imply otherwise."""
    assert "eidomira_studio_prefs" in APP_JS
    assert "stored in this browser only" in view("Settings").lower()


def test_only_a_session_failure_can_end_a_session():
    """Three ways this went wrong, each reported by the user as "it logged me out".

    Any transient error inside the account load used to clear the token, so a rendering
    failure looked like a logout. A wrong current password used to answer 401 and sign the
    user out of the page they were using to change it. And any 401 from *any* endpoint used to
    end the session — `/api/billing/account` answers 401 to an anonymous caller by design, and
    the page read that as a dead session while `/api/auth/me` was answering 200 with the
    account. Only the endpoint that identifies a token may end a session now, and it is asked
    before anything is ended.
    """
    # The session check is its own call, routed around apiFetch so it cannot answer circularly,
    # and it is the only one whose failure ends a session.
    assert "async function sessionUser()" in APP_JS
    assert "apiFetch('/api/auth/me')" not in APP_JS, "the session check goes through the caller"
    assert "endSession(" in APP_JS
    assert "location.replace" in APP_JS, "nothing sends a visitor to the sign-in card"

    # A 401 buys a restore, a retry, and then a question put to the session endpoint — in that
    # order, and only then a decision.
    api = APP_JS[APP_JS.index("async function apiFetch"):APP_JS.index("async function sessionUser()")]
    assert api.index("restoreDemoSession") < api.index("await sessionUser()")
    assert api.index("await sessionUser()") < api.index("endSession(")
    assert "user===undefined" in api, "a server that did not answer is treated as a sign-out"

    # The account load tells the three outcomes apart instead of collapsing them.
    load = APP_JS[APP_JS.index("async function loadAccount()"):]
    assert "if(user===undefined)" in load and "if(user===null)" in load
    assert "if(!user)return;" not in load, "an unknown answer is treated as no session"

    # And the catch that used to sign people out keeps the session.
    assert "catch{setToken('');location.replace('/?signin=1')}" not in APP_JS
    assert "Your account did not finish loading" in APP_JS

    # The password form is still the one call that says a 401 is not about the session.
    assert "reauth:false" in APP_JS
    password_call = APP_JS[APP_JS.index("'/api/auth/password'"):]
    assert password_call[:200].count("reauth:false") == 1


def test_an_ended_session_says_so():
    """Being dropped on a bare sign-in card is indistinguishable from a broken login, which
    is how it was reported."""
    assert "ended=1" in APP_JS
    for script, name in ((APP_JS, "app.js"), ((STATIC / "landing.js").read_text(encoding="utf-8"),
                                             "landing.js")):
        assert "session ended" in script, f"{name} stays silent about it"
    assert 'id="authMessage"' in APP_HTML, "nowhere to say it"


def test_a_200_from_the_session_endpoint_is_not_by_itself_a_session():
    """`/api/auth/me` answers 200 to an anonymous caller, with a guest record. A caller that
    only checked `response.ok` took that guest for a signed-in user, which is why the one
    function whose job is to decide the session could not decide it."""
    session = APP_JS[APP_JS.index("async function sessionUser()"):APP_JS.index("window.eidomiraSession")]
    assert "response.status===401)return null" in session, "a refused token is not recognised"
    assert "role==='guest'" in session, "the guest record is taken for a user"
    assert "local-guest" in session, "the guest record is taken for a user"
    assert "return undefined" in session, "the server not answering is read as a sign-out"


def test_a_session_the_server_cannot_see_is_tried_for_before_it_is_ended():
    """The third report, and the shape of it: the studio was bounced to `/?signin=1&ended=1`
    with no 401 anywhere in the access log — `/api/auth/me` answered 200, as a *guest*, so no
    credential had reached the server even though the page was holding a token. From the
    client those two cases are the same response, so the only safe reading is the recoverable
    one: try to put a session back, re-ask, and end it only if the ask says so."""
    load = APP_JS[APP_JS.index("async function loadAccount()"):]
    assert "if(await restoreDemoSession()) user=await sessionUser()" in load, \
        "a session the server cannot see ends without being tried for"
    assert load.index("restoreDemoSession") < load.index("endSession(")
    # And the reason it can be a guest with a token in hand is said out loud.
    assert "no credential reached it" in APP_JS


def test_the_console_can_be_asked_what_the_page_believes():
    """The user asked for the console. Every one of these failures is a decision this file
    makes, and that decision was invisible from outside, so it can be asked for by name."""
    assert "window.eidomiraSession" in APP_JS
    report = APP_JS[APP_JS.index("window.eidomiraSession"):]
    assert "token:accessToken?'present':'none'" in report, "it does not say whether a token is held"
    assert "server:" in report, "it does not say what the server thinks"
    assert "accessToken}" not in report, "it would print the credential itself"


def test_the_engine_card_states_what_the_backend_is():
    assert "reportDiagnostics" in APP_JS
    assert "diagnostic backend" in APP_JS, "the diagnostic backend is not named anywhere"


# ── the studio keeps its contract ────────────────────────────────────────────

def test_the_studio_view_keeps_the_workbench_and_its_steps():
    studio = view("Studio")
    for element in ("studioSection", "source", "enrollBtn", "video", "output", "goBtn",
                    "recordBtn", "cleanBtn", "consensus" if False else "consent",
                    "telemetry", "trainer", "callDock"):
        assert f'id="{element}"' in studio or f'class="{element}"' in studio, element
    assert 'class="progress"' in studio, "the 01/02/03 strip left the studio"
    assert 'class="canvases"' in studio


def test_the_settings_cards_are_styled():
    """A page of unstyled markup is not a rebuild. Each new structural class needs a rule."""
    for hook in (".tabs", ".tab", ".view", ".card", ".billing", ".settings", ".packCard",
                 ".ledger", ".facts", ".dash", ".accountChip", "body.reduceMotion"):
        assert re.search(rf"{re.escape(hook)}[^{{}}]*\{{", STUDIO_CSS), hook
