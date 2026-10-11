"""Buying credits, and the balance a customer is shown afterwards.

The failure this file exists for is the one that costs both money and trust: a customer
pays and the credits never appear. It was real here. Checkout sent the payer back to
`/?payment=return`, and only the studio — at `/app` — knew how to verify a reference. On an
installation whose Paystack webhook is not configured, which is every installation until
somebody configures one, the money arrived and the balance did not move.

So these tests cover the whole path, not the pieces: the price on the button is the amount
Paystack is asked for, the payment lands the customer on a page that can verify it, the
credits are granted exactly once however many times Paystack repeats itself, and a ledger
shows one account's movements to that account only.

The network call is the one thing that cannot be made from here, and it is faked openly:
`_request` is replaced, and the amount it receives is asserted against the catalogue, so a
price change that never reached checkout still fails.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app import paystack as paystack_module
from app.billing import TOPUPS, account as billing_account, consume, topup_catalogue
from app.config import settings
from app.database import database
from app.main import app
from app.security import issue_email_token, register as register_account, verify_email_token

PASSWORD = "a-sufficiently-long-passphrase"
TOPUP_REFERENCE_PREFIX = "paystack:"


@pytest.fixture(autouse=True)
def _always_isolated(isolated_db):
    yield


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def paystack(monkeypatch):
    """A Paystack that answers instantly and records what it was asked."""
    monkeypatch.setattr(settings, "paystack_secret_key", "sk_test_example")
    state: dict = {"calls": [], "verify": {}}

    def fake_request(path, payload=None):
        state["calls"].append((path, payload))
        if path == "/transaction/initialize":
            return {"authorization_url": "https://checkout.paystack.com/abc123",
                    "access_code": "ac_abc123"}
        if path.startswith("/transaction/verify/"):
            return state["verify"]
        raise AssertionError(f"unexpected Paystack call: {path}")

    monkeypatch.setattr(paystack_module, "_request", fake_request)
    return state


def verified_account(client, email: str) -> dict:
    """Register, verify the address, sign in — the state a customer is in before paying."""
    user = register_account(email, PASSWORD)
    verify_email_token(issue_email_token(user["id"]))
    response = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200
    return response.json()["user"]


def balance(user_id: str) -> int:
    return billing_account(user_id)["wallet"]["total"]


def charge(reference: str, amount_kobo: int, *, status: str = "success",
           currency: str = "NGN") -> dict:
    """The shape Paystack's verify endpoint returns, narrowed to the fields we act on."""
    return {"status": status, "reference": reference, "amount": amount_kobo,
            "currency": currency, "customer": {"email": "buyer@example.com"}}


def buy(client, product: str) -> str:
    response = client.post("/api/payments/paystack/checkout", json={"product": product})
    assert response.status_code == 200, response.text
    return response.json()["reference"]


def webhook(client, event: dict) -> int:
    raw = json.dumps(event).encode()
    signature = hmac.new(settings.paystack_secret_key.encode(), raw, hashlib.sha512).hexdigest()
    return client.post("/api/payments/paystack/webhook", content=raw,
                       headers={"x-paystack-signature": signature}).status_code


# ── the price shown is the price charged ──────────────────────────────────────

def test_the_price_on_the_button_is_the_price_paystack_is_asked_for(client, paystack):
    """The packs used to be typed into the page by hand, separately from the settings that
    decide the charge. They agreed by luck; this is the assertion that they agree by
    construction."""
    verified_account(client, "buyer@example.com")
    catalogue = client.get("/api/billing/account").json()["topups"]
    assert {pack["product"] for pack in catalogue} == set(TOPUPS)

    for pack in catalogue:
        shown = int(pack["price"].lstrip("\u20a6").replace(",", ""))
        assert shown * 100 == pack["amount_kobo"], "the label and the charge disagree"

        reference = buy(client, pack["product"])
        assert reference.startswith("eid_")
        path, payload = paystack["calls"][-1]
        assert path == "/transaction/initialize"
        assert payload["amount"] == pack["amount_kobo"]
        assert payload["currency"] == "NGN"
        assert payload["metadata"]["product"] == pack["product"]
        assert payload["metadata"]["kind"] == "topup"


def test_checkout_returns_the_customer_to_a_page_that_can_verify_it(client, paystack):
    """The bug this file was written for. `/` has no account context and no verification
    step, so a payer sent there waits for a webhook that may not exist."""
    verified_account(client, "buyer@example.com")
    buy(client, "credits-200")

    _, payload = paystack["calls"][-1]
    assert payload["callback_url"].endswith("/app?payment=return"), payload["callback_url"]
    assert payload["callback_url"].startswith(settings.public_url.rstrip("/"))


def test_checkout_needs_a_verified_address(client, paystack):
    """Paystack is handed the address, so an unconfirmed one would be billed to an address
    nobody has proved they can receive mail at."""
    register_account("unverified@example.com", PASSWORD)
    client.post("/api/auth/login", json={"email": "unverified@example.com", "password": PASSWORD})

    response = client.post("/api/payments/paystack/checkout", json={"product": "credits-200"})

    assert response.status_code == 403
    assert "Verify your email" in response.json()["error"]
    assert paystack["calls"] == [], "a refused checkout must not reach Paystack"


def test_checkout_is_anonymous_hostile(client):
    assert client.post("/api/payments/paystack/checkout",
                       json={"product": "credits-200"}).status_code == 401


def test_checkout_says_so_when_paystack_is_not_configured(client, isolated_db):
    """Rather than a 500 or, worse, a button that appears to work."""
    user = register_account("buyer@example.com", PASSWORD)
    verify_email_token(issue_email_token(user["id"]))
    client.post("/api/auth/login", json={"email": "buyer@example.com", "password": PASSWORD})

    response = client.post("/api/payments/paystack/checkout", json={"product": "credits-200"})

    assert response.status_code == 503
    assert "not configured" in response.json()["error"]
    assert settings.paystack_secret_key == "", "this test is meaningless if a key is set"


def test_an_unknown_product_is_refused(client, paystack):
    verified_account(client, "buyer@example.com")
    assert client.post("/api/payments/paystack/checkout",
                       json={"product": "credits-999999"}).status_code == 400


# ── the money becomes credits, once ──────────────────────────────────────────

def test_a_paid_topup_credits_the_balance(client, paystack):
    user = verified_account(client, "buyer@example.com")
    assert balance(user["id"]) == 0

    reference = buy(client, "credits-500")
    paystack["verify"] = charge(reference, TOPUPS["credits-500"][1]())

    response = client.get(f"/api/payments/paystack/verify/{reference}")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["kind"] == "topup", "the page confirms what was bought"
    assert balance(user["id"]) == 500

    shown = client.get("/api/billing/account").json()
    assert shown["wallet"]["topup_credits"] == 500
    assert shown["wallet"]["total"] == 500
    assert shown["credits_used"] == 0
    assert shown["ledger"][0]["delta"] == 500
    assert shown["ledger"][0]["balance_after"] == 500


def test_paystack_repeating_itself_does_not_pay_twice(client, paystack):
    """Verify runs on the return trip and the webhook runs whenever Paystack feels like it,
    so the same charge is seen at least twice. The ledger reference is what makes the
    second sighting a no-op."""
    user = verified_account(client, "buyer@example.com")
    reference = buy(client, "credits-200")
    amount = TOPUPS["credits-200"][1]()
    paystack["verify"] = charge(reference, amount)

    assert client.get(f"/api/payments/paystack/verify/{reference}").status_code == 200
    assert client.get(f"/api/payments/paystack/verify/{reference}").status_code == 200
    assert balance(user["id"]) == 200

    event = {"event": "charge.success",
             "data": {**charge(reference, amount), "metadata": {"eidomira_user_id": user["id"],
                                                               "product": "credits-200",
                                                               "kind": "topup"}}}
    assert webhook(client, event) == 200
    assert webhook(client, event) == 200, "a redelivery is accepted, not acted on"
    assert balance(user["id"]) == 200, "the customer was credited more than once"

    granted = database.one("SELECT COUNT(*) AS n FROM credit_ledger WHERE reference=?",
                           (TOPUP_REFERENCE_PREFIX + reference,))
    assert granted["n"] == 1


def test_an_amount_that_does_not_match_the_intent_is_not_credited(client, paystack):
    """Belt and braces between what we asked for and what was charged. A mismatch means one
    of the two is wrong, and neither answer is "grant the credits anyway"."""
    user = verified_account(client, "buyer@example.com")
    reference = buy(client, "credits-200")
    paystack["verify"] = charge(reference, 100)  # ₦1, not ₦12,000

    response = client.get(f"/api/payments/paystack/verify/{reference}")

    assert response.status_code == 200 and response.json()["status"] == "success"
    assert balance(user["id"]) == 0, "credits were granted for an amount we never charged"


@pytest.mark.parametrize("bad", [
    {"status": "failed"},
    {"currency": "USD"},
])
def test_a_charge_that_did_not_succeed_is_not_credited(client, paystack, bad):
    user = verified_account(client, "buyer@example.com")
    reference = buy(client, "credits-200")
    paystack["verify"] = {**charge(reference, TOPUPS["credits-200"][1]()), **bad}

    response = client.get(f"/api/payments/paystack/verify/{reference}")

    assert response.status_code == 200
    assert response.json()["status"] == bad.get("status", "success")
    assert balance(user["id"]) == 0


def test_the_balance_goes_down_when_credits_are_spent(client, paystack):
    """What the customer is shown afterwards is the point of the panel: a balance that only
    ever goes up is not a balance."""
    user = verified_account(client, "buyer@example.com")
    reference = buy(client, "credits-500")
    paystack["verify"] = charge(reference, TOPUPS["credits-500"][1]())
    client.get(f"/api/payments/paystack/verify/{reference}")

    consume(user["id"], "voice_changer", 10)  # 5 credits a minute

    shown = client.get("/api/billing/account").json()
    assert shown["wallet"]["total"] == 500 - 50
    assert shown["credits_used"] == 50
    assert shown["ledger"][0]["delta"] == -50
    assert shown["ledger"][0]["balance_after"] == 450


# ── the ledger belongs to one account ────────────────────────────────────────

def test_a_ledger_shows_only_its_own_accounts_movements(client, paystack):
    """Read inside the query rather than filtered afterwards: a ledger that is fetched and
    then filtered is one forgotten condition away from showing somebody else's purchases."""
    first = verified_account(client, "first@example.com")
    first_reference = buy(client, "credits-200")
    paystack["verify"] = charge(first_reference, TOPUPS["credits-200"][1]())
    client.get(f"/api/payments/paystack/verify/{first_reference}")

    client.post("/api/auth/logout")
    second = verified_account(client, "second@example.com")
    second_reference = buy(client, "credits-500")
    paystack["verify"] = charge(second_reference, TOPUPS["credits-500"][1]())
    client.get(f"/api/payments/paystack/verify/{second_reference}")

    mine = client.get("/api/billing/account").json()["ledger"]
    references = {entry["reference"] for entry in mine}

    assert TOPUP_REFERENCE_PREFIX + second_reference in references
    assert TOPUP_REFERENCE_PREFIX + first_reference not in references
    assert balance(second["id"]) == 500
    assert balance(first["id"]) == 200


def test_the_billing_account_is_not_readable_without_a_session(client):
    assert client.get("/api/billing/account").status_code == 401


def test_the_catalogue_is_not_empty_and_prices_are_whole_naira():
    """A pack priced at a fraction of a naira would be a bug in kobo arithmetic, and a pack
    the page cannot price would render a blank button."""
    catalogue = topup_catalogue()
    assert catalogue
    for pack in catalogue:
        assert pack["amount_kobo"] % 100 == 0, f"{pack['product']} is not a whole naira"
        assert pack["price"].startswith("\u20a6")
        assert pack["credits"] > 0
