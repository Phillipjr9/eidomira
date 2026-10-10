from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
import uuid
from app.config import settings
from app.database import database
from app.billing import PRO_CREDITS, TOPUPS

API="https://api.paystack.co"


def _request(path: str, payload: dict | None=None):
    if not settings.paystack_secret_key: raise RuntimeError("Paystack is not configured")
    body=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(API+path,data=body,method="POST" if body is not None else "GET",headers={
        "Authorization":f"Bearer {settings.paystack_secret_key}","Content-Type":"application/json","Accept":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=20) as response: result=json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try: message=json.loads(exc.read()).get("message","Paystack request failed")
        except Exception: message="Paystack request failed"
        raise RuntimeError(message) from exc
    if not result.get("status"): raise RuntimeError(result.get("message","Paystack request failed"))
    return result["data"]


def checkout(user: dict, product: str):
    if not user.get("email_verified"): raise PermissionError("Verify your email before checkout")
    reference="eid_"+uuid.uuid4().hex
    if product == "live-pro-monthly":
        amount=settings.paystack_monthly_amount_kobo; plan=settings.paystack_monthly_plan_code; kind="subscription"
    elif product == "live-pro-annual":
        amount=settings.paystack_annual_amount_kobo; plan=settings.paystack_annual_plan_code; kind="subscription"
    elif product in TOPUPS:
        amount=TOPUPS[product][1](); plan=""; kind="topup"
    else: raise ValueError("Unknown Paystack product")
    if kind == "subscription" and not plan: raise RuntimeError("Create the Paystack plan and configure its plan code first")
    metadata={"eidomira_user_id":user["id"],"product":product,"kind":kind}
    payload={"email":user["email"],"amount":amount,"currency":"NGN","reference":reference,
             # Back to the studio, not the landing page. The landing page has no
             # account context and no way to verify a reference, so a customer sent
             # there paid and then waited for a webhook — which, unconfigured, never
             # came. tests/test_credits.py pins this URL.
             "callback_url":f"{settings.public_url.rstrip('/')}/app?payment=return","metadata":metadata}
    if plan: payload["plan"]=plan
    data=_request("/transaction/initialize",payload)
    database.execute("INSERT INTO payment_intents(reference,user_id,kind,product,amount_kobo,status,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                     (reference,user["id"],kind,product,amount,"initialized",int(time.time()),json.dumps(metadata)))
    return {"reference":reference,"authorization_url":data["authorization_url"],"access_code":data.get("access_code"),"product":product,"amount_kobo":amount,"currency":"NGN"}


def verify_transaction(reference: str, user_id: str | None=None):
    if not reference.startswith("eid_"): raise ValueError("Invalid reference")
    intent=database.one("SELECT * FROM payment_intents WHERE reference=?",(reference,))
    if not intent or (user_id and intent["user_id"] != user_id): raise ValueError("Payment not found")
    data=_request("/transaction/verify/"+reference)
    if data.get("status") == "success":
        _fulfil_charge(data)
    return {"reference":reference,"status":data.get("status"),"amount":data.get("amount"),"currency":data.get("currency"),
            # So the page can confirm the thing that was actually bought. "Live Pro is
            # active" is the wrong sentence to show somebody who just bought credits.
            "kind":intent["kind"] if intent else None,"product":intent["product"] if intent else None}


def valid_signature(raw: bytes, signature: str | None):
    if not settings.paystack_secret_key or not signature: return False
    expected=hmac.new(settings.paystack_secret_key.encode(),raw,hashlib.sha512).hexdigest()
    return hmac.compare_digest(expected,signature)


def process_webhook(raw: bytes):
    event=json.loads(raw); event_key=hashlib.sha256(raw).hexdigest(); received=int(time.time())
    with database.lock, database.connect() as db:
        if db.execute("SELECT 1 FROM payment_events WHERE event_key=?",(event_key,)).fetchone(): return {"duplicate":True}
        db.execute("INSERT INTO payment_events(event_key,event_type,received_at,payload_json) VALUES(?,?,?,?)",
                   (event_key,event.get("event","unknown"),received,raw.decode()))
    kind=event.get("event"); data=event.get("data") or {}
    if kind == "charge.success": _fulfil_charge(data)
    elif kind in {"subscription.disable","subscription.not_renew"}: _subscription_status(data,"cancelled" if kind.endswith("disable") else "non-renewing")
    database.execute("UPDATE payment_events SET processed_at=? WHERE event_key=?",(int(time.time()),event_key))
    return {"processed":True,"event":kind}


def _user_from(data: dict, intent=None):
    if intent: return intent["user_id"]
    metadata=data.get("metadata") or {}
    if isinstance(metadata,str):
        try: metadata=json.loads(metadata)
        except Exception: metadata={}
    if metadata.get("eidomira_user_id"): return metadata["eidomira_user_id"]
    email=(data.get("customer") or {}).get("email")
    user=database.one("SELECT id FROM users WHERE email=? AND email_verified_at IS NOT NULL",(email.lower(),)) if email else None
    return user["id"] if user else None


def _fulfil_charge(data: dict):
    if data.get("status") != "success" or data.get("currency") != "NGN": return
    reference=str(data.get("reference") or "")
    if not reference: return
    intent=database.one("SELECT * FROM payment_intents WHERE reference=?",(reference,))
    if intent and (data.get("amount") != intent["amount_kobo"]): return
    if database.one("SELECT 1 AS ok FROM credit_ledger WHERE reference=?",("paystack:"+reference,)): return
    user_id=_user_from(data,intent)
    if not user_id: return
    metadata=data.get("metadata") or {}
    if isinstance(metadata,str):
        try: metadata=json.loads(metadata)
        except Exception: metadata={}
    product=intent["product"] if intent else metadata.get("product")
    if not product:
        plan_data=data.get("plan") or {}
        plan_code=plan_data.get("plan_code") if isinstance(plan_data,dict) else plan_data
        if plan_code == settings.paystack_annual_plan_code: product="live-pro-annual"
        elif plan_code == settings.paystack_monthly_plan_code: product="live-pro-monthly"
        else: return
    if not intent:
        expected=settings.paystack_annual_amount_kobo if product.endswith("annual") else settings.paystack_monthly_amount_kobo
        if data.get("amount") != expected: return
    if product in TOPUPS: _grant_topup(user_id,TOPUPS[product][0],reference,product)
    elif product in {"live-pro-monthly","live-pro-annual"}: _activate_subscription(user_id,reference,product,data)
    if intent: database.execute("UPDATE payment_intents SET status='completed',completed_at=? WHERE reference=?",(int(time.time()),reference))


def _activate_subscription(user_id: str, reference: str, product: str, data: dict):
    current=int(time.time()); annual=product.endswith("annual"); period=current+(365 if annual else 31)*86400
    credits=PRO_CREDITS*12 if annual else PRO_CREDITS
    customer=(data.get("customer") or {}).get("customer_code"); subscription=(data.get("subscription") or {}).get("subscription_code") if isinstance(data.get("subscription"),dict) else data.get("subscription")
    with database.lock, database.connect() as db:
        db.execute("UPDATE subscriptions SET status='superseded' WHERE user_id=? AND status IN ('trialing','active')",(user_id,))
        db.execute("INSERT INTO subscriptions(id,user_id,plan,status,current_period_start,current_period_end,provider,provider_customer_id,provider_subscription_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (uuid.uuid4().hex,user_id,"live-pro-annual" if annual else "live-pro","active",current,period,"paystack",customer,subscription,current))
        wallet=db.execute("SELECT topup_credits FROM credit_wallets WHERE user_id=?",(user_id,)).fetchone(); topup=wallet[0] if wallet else 0
        db.execute("INSERT INTO credit_wallets(user_id,subscription_credits,topup_credits,updated_at) VALUES(?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET subscription_credits=excluded.subscription_credits,updated_at=excluded.updated_at",
                   (user_id,credits,topup,current))
        db.execute("INSERT INTO credit_ledger(id,user_id,delta,balance_after,kind,description,reference,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
                   (uuid.uuid4().hex,user_id,credits,credits+topup,"subscription_grant","Paystack Live Pro renewal","paystack:"+reference,current,json.dumps({"product":product})))


def _grant_topup(user_id: str, credits: int, reference: str, product: str):
    current=int(time.time())
    with database.lock, database.connect() as db:
        wallet=db.execute("SELECT subscription_credits,topup_credits,topup_expires_at FROM credit_wallets WHERE user_id=?",(user_id,)).fetchone(); sub=wallet[0] if wallet else 0
        active_top=wallet[1] if wallet and wallet[2] and wallet[2]>current else 0; new_top=active_top+credits; expires=current+365*86400
        db.execute("INSERT INTO credit_wallets(user_id,subscription_credits,topup_credits,topup_expires_at,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET topup_credits=excluded.topup_credits,topup_expires_at=excluded.topup_expires_at,updated_at=excluded.updated_at",(user_id,sub,new_top,expires,current))
        db.execute("INSERT INTO credit_ledger(id,user_id,delta,balance_after,kind,description,reference,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
                   (uuid.uuid4().hex,user_id,credits,sub+new_top,"topup","Paystack credit top-up","paystack:"+reference,current,json.dumps({"product":product})))


def _subscription_status(data: dict, status: str):
    code=data.get("subscription_code") or (data.get("subscription") or {}).get("subscription_code")
    if code: database.execute("UPDATE subscriptions SET status=? WHERE provider='paystack' AND provider_subscription_id=?",(status,code))
