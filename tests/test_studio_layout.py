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


def test_a_dead_session_is_noticed_instead_of_being_ignored():
    """The user's own log showed it: /api/auth/me answered 200 (the guest fallback) while
    /api/billing/account and four checkout attempts answered 401, because the browser held a
    token for an account the rebuilt database no longer had. The page looked signed in and did
    nothing. One place now treats a 401 on a request that carried a token as the session being
    over."""
    assert "response.status===401&&accessToken" in APP_JS
    assert "setToken('');location.replace('/?signin=1')" in APP_JS


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
