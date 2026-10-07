from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from app.config import settings
from app.database import database

TRIAL_DAYS = 7
TRIAL_CREDITS = 100
PRO_CREDITS = 1500

PLAN = {
    "id": "live-pro",
    "name": "Eidomira Live Pro",
    "price_usd": 39.99,
    "annual_price_usd": 399,
    "price_ngn": 59900,
    "annual_price_ngn": 599000,
    "payment_provider": "paystack",
    "monthly_credits": PRO_CREDITS,
    "features": [
        "10 minutes/month Full Live Swap", "60 minutes/day Voice Changer",
        "180-second videos", "500 MB uploads", "HD output", "Face Swap",
        "Character Swap", "Lip Sync", "Realistic Talking Avatar", "Batch Processing",
        "Priority Processing", "Metered API Access", "Voice Cloner", "No promotional watermark"
    ],
}

#: Credit top-ups: product id -> (credits, how to read the price from settings).
#: Defined here rather than beside the payment provider because the credits are the
#: product and Paystack is only how it is paid for; `app/paystack.py` imports this so the
#: amount it asks Paystack for and the amount the button offers are the same number.
TOPUPS = {
    "credits-200": (200, lambda: settings.paystack_topup_200_kobo),
    "credits-500": (500, lambda: settings.paystack_topup_500_kobo),
    "credits-1500": (1500, lambda: settings.paystack_topup_1500_kobo),
    "credits-5000": (5000, lambda: settings.paystack_topup_5000_kobo),
}

TOOLS = {
    "live_swap": {"name":"Full Live Swap", "unit":"minute", "credits":100, "max":10, "trial":False},
    "voice_changer": {"name":"Voice Changer", "unit":"minute", "credits":5, "daily_free":60, "trial":True},
    "face_swap_hd": {"name":"HD Face Swap", "unit":"second", "credits":6, "max":180, "trial":True},
    "character_swap_hd": {"name":"HD Character Swap", "unit":"second", "credits":7, "max":180, "trial":False},
    "lip_sync_hd": {"name":"HD Lip Sync", "unit":"second", "credits":4, "max":180, "trial":True},
    "talking_avatar_hd": {"name":"Realistic Talking Avatar", "unit":"second", "credits":8, "max":180, "trial":False},
    "voice_clone_profile": {"name":"Voice Clone Profile", "unit":"profile", "credits":200, "max":5, "trial":False},
    "voice_clone_speech": {"name":"Cloned Voice Speech", "unit":"minute", "credits":4, "trial":False},
}


def now() -> int: return int(time.time())


def topup_catalogue() -> list[dict]:
    """The packs as a customer sees them, priced from the settings the charge uses.

    The option labels used to be typed into the page by hand and the amounts live in
    settings, so the two only agreed by luck. Anything that displays a price reads it
    here.
    """
    packs = []
    for product, (credits, amount_of) in TOPUPS.items():
        kobo = amount_of()
        packs.append({"product": product, "credits": credits, "amount_kobo": kobo,
                      "price": "\u20a6" + f"{kobo // 100:,}"})
    return packs


def ledger(user_id: str, limit: int = 20) -> list[dict]:
    """This account's own credit movements, newest first.

    Scoped by user id in the WHERE clause rather than filtered afterwards: a ledger that
    is read and then filtered is one forgotten condition away from showing somebody else's
    purchases.
    """
    with database.lock, database.connect() as db:
        rows = db.execute(
            "SELECT delta,balance_after,kind,description,reference,created_at FROM credit_ledger "
            "WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?", (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def create_trial(user_id: str):
    created = now(); ends = created + TRIAL_DAYS * 86400
    with database.lock, database.connect() as db:
        if db.execute("SELECT 1 FROM subscriptions WHERE user_id=?", (user_id,)).fetchone(): return
        db.execute("INSERT INTO subscriptions(id,user_id,plan,status,current_period_start,current_period_end,created_at) VALUES(?,?,?,?,?,?,?)",
                   (uuid.uuid4().hex,user_id,"trial","trialing",created,ends,created))
        db.execute("INSERT INTO credit_wallets(user_id,subscription_credits,topup_credits,updated_at) VALUES(?,?,0,?)",
                   (user_id,TRIAL_CREDITS,created))
        db.execute("INSERT INTO credit_ledger(id,user_id,delta,balance_after,kind,description,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                   (uuid.uuid4().hex,user_id,TRIAL_CREDITS,TRIAL_CREDITS,"trial_grant","Verified-email trial",created,"{}"))


def account(user_id: str):
    sub = database.one("SELECT plan,status,current_period_start,current_period_end FROM subscriptions WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (user_id,))
    current=now()
    wallet = database.one("SELECT subscription_credits,topup_credits,topup_expires_at,updated_at FROM credit_wallets WHERE user_id=?", (user_id,)) or {"subscription_credits":0,"topup_credits":0,"topup_expires_at":None,"updated_at":current}
    subscription_active=bool(sub and sub["status"] in {"trialing","active"} and sub["current_period_end"]>current)
    subscription_available=wallet["subscription_credits"] if subscription_active else 0
    topup_available=wallet["topup_credits"] if wallet["topup_expires_at"] and wallet["topup_expires_at"]>current else 0
    usage = database.one("SELECT COALESCE(SUM(-delta),0) AS used FROM credit_ledger WHERE user_id=? AND delta<0", (user_id,))
    return {"subscription":sub,"wallet":{**wallet,"subscription_credits":subscription_available,"topup_credits":topup_available,"total":subscription_available+topup_available},"credits_used":usage["used"],"plan":PLAN,"tools":TOOLS,"payg":True,
            # Everything the credits panel needs, in the payload it already fetches:
            # the packs with their real prices, this account's own history, and whether
            # card payments are switched on at all, so the page can say so instead of
            # offering four buttons that fail.
            "topups":topup_catalogue(),"ledger":ledger(user_id),
            "paystack_configured":bool(settings.paystack_secret_key)}


def quote(tool: str, quantity: float, api: bool=False):
    item=TOOLS.get(tool)
    if not item: raise ValueError("Unknown tool")
    if quantity <= 0: raise ValueError("Quantity must be positive")
    if item.get("max") and quantity > item["max"]: raise ValueError(f"Maximum is {item['max']} {item['unit']}s")
    cost=max(1, int(quantity*item["credits"] + .9999))
    if api: cost=max(1, int(cost*1.25 + .9999))
    return {"tool":tool,"quantity":quantity,"unit":item["unit"],"credits":cost,"api_multiplier":1.25 if api else 1}


def consume(user_id: str, tool: str, quantity: float, *, api=False, reference=None):
    charge=quote(tool,quantity,api); created=now()
    with database.lock, database.connect() as db:
        user=db.execute("SELECT email_verified_at FROM users WHERE id=?",(user_id,)).fetchone()
        if not user or not user[0]: raise PermissionError("Verify your email before using credits")
        sub=db.execute("SELECT plan,status,current_period_end FROM subscriptions WHERE user_id=? ORDER BY created_at DESC LIMIT 1",(user_id,)).fetchone()
        subscription_active=bool(sub and sub[1] in {"trialing","active"} and sub[2]>created)
        wallet=db.execute("SELECT subscription_credits,topup_credits,topup_expires_at FROM credit_wallets WHERE user_id=?",(user_id,)).fetchone()
        sub_credits=wallet[0] if wallet and subscription_active else 0
        topup_credits=wallet[1] if wallet and wallet[2] and wallet[2]>created else 0
        if not subscription_active and not topup_credits: raise PermissionError("An active plan, trial, or PAYG credit balance is required")
        if subscription_active and sub[0] == "trial" and not TOOLS[tool].get("trial") and not topup_credits: raise PermissionError("This tool requires Live Pro or PAYG credits")
        if api and (not subscription_active or sub[0] not in {"live-pro","live-pro-annual"}): raise PermissionError("API access requires Live Pro")
        total=sub_credits+topup_credits
        if total < charge["credits"]: raise PermissionError("Not enough credits")
        # Trial/subscription credits are consumed first for eligible tools; PAYG covers the remainder.
        can_use_sub=not (sub and sub[0] == "trial" and not TOOLS[tool].get("trial"))
        from_sub=min(sub_credits,charge["credits"]) if can_use_sub else 0; from_top=charge["credits"]-from_sub
        new_sub=(wallet[0]-from_sub) if wallet else 0; new_top=(wallet[1]-from_top) if wallet else 0; balance=(new_sub if subscription_active else 0)+new_top
        db.execute("UPDATE credit_wallets SET subscription_credits=?,topup_credits=?,updated_at=? WHERE user_id=?",(new_sub,new_top,created,user_id))
        db.execute("INSERT INTO credit_ledger(id,user_id,delta,balance_after,kind,description,reference,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
                   (uuid.uuid4().hex,user_id,-charge["credits"],balance,"usage",TOOLS[tool]["name"],reference,created,json.dumps(charge)))
    return {**charge,"balance":balance}
