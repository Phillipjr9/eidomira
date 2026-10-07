from __future__ import annotations

import re
import time
import uuid
import secrets
import hashlib
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Header, HTTPException, Cookie

from app.config import settings
from app.database import database

EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)


def register(email: str, password: str):
    email = email.strip().lower()
    if not EMAIL.fullmatch(email): raise ValueError("Enter a valid email address")
    if len(password) < 10: raise ValueError("Password must contain at least 10 characters")
    user_id = uuid.uuid4().hex
    try:
        database.execute("INSERT INTO users(id,email,password_hash,created_at) VALUES(?,?,?,?)",
                         (user_id, email, hasher.hash(password), int(time.time())))
    except Exception as exc:
        if "UNIQUE" in str(exc): raise ValueError("An account already uses that email") from exc
        raise
    return {"id": user_id, "email": email, "email_verified": False}


def issue_email_token(user_id: str):
    raw=secrets.token_urlsafe(32); digest=hashlib.sha256(raw.encode()).hexdigest(); created=int(time.time())
    database.execute("UPDATE email_verification_tokens SET used_at=? WHERE user_id=? AND used_at IS NULL",(created,user_id))
    database.execute("INSERT INTO email_verification_tokens(id,user_id,token_hash,expires_at,created_at) VALUES(?,?,?,?,?)",
                     (uuid.uuid4().hex,user_id,digest,created+settings.email_token_ttl,created))
    return raw


def verify_email_token(raw: str):
    digest=hashlib.sha256(raw.encode()).hexdigest(); current=int(time.time())
    token=database.one("SELECT * FROM email_verification_tokens WHERE token_hash=? AND used_at IS NULL",(digest,))
    if not token or token["expires_at"] < current: return None
    with database.lock, database.connect() as db:
        db.execute("UPDATE email_verification_tokens SET used_at=? WHERE id=? AND used_at IS NULL",(current,token["id"]))
        db.execute("UPDATE users SET email_verified_at=COALESCE(email_verified_at,?) WHERE id=?",(current,token["user_id"]))
        user=db.execute("SELECT id,email,email_verified_at FROM users WHERE id=?",(token["user_id"],)).fetchone()
    return {"id":user[0],"email":user[1],"email_verified":bool(user[2])}


def authenticate(email: str, password: str):
    user = database.one("SELECT * FROM users WHERE email=? AND disabled=0", (email.strip().lower(),))
    if not user: return None
    try: hasher.verify(user["password_hash"], password)
    except VerifyMismatchError: return None
    if hasher.check_needs_rehash(user["password_hash"]):
        database.execute("UPDATE users SET password_hash=? WHERE id=?", (hasher.hash(password), user["id"]))
    return {"id": user["id"], "email": user["email"], "email_verified":bool(user.get("email_verified_at"))}


def access_token(user):
    now = int(time.time())
    return jwt.encode({"sub":user["id"],"email":user["email"],"iat":now,"nbf":now-2,
                       "exp":now+settings.access_token_ttl,"typ":"access"},
                      settings.auth_secret, algorithm="HS256")


def optional_user(authorization: str | None = Header(default=None), eidomira_access_token: str | None = Cookie(default=None)):
    if not authorization and not eidomira_access_token:
        if settings.require_auth: raise HTTPException(401, "Authentication required")
        return {"id":"local-guest","email":"local@eidomira.invalid","email_verified":False}
    try:
        if authorization:
            scheme, token = authorization.split(" ",1)
            if scheme.lower() != "bearer": raise ValueError
        else:
            token=eidomira_access_token
        payload = jwt.decode(token, settings.auth_secret, algorithms=["HS256"])
        user = database.one("SELECT id,email,disabled,email_verified_at FROM users WHERE id=?", (payload["sub"],))
        if not user or user["disabled"]: raise ValueError
        return {"id":user["id"],"email":user["email"],"email_verified":bool(user["email_verified_at"])}
    except Exception as exc:
        raise HTTPException(401, "Invalid or expired access token") from exc


def authenticated_user(authorization: str | None = Header(default=None), eidomira_access_token: str | None = Cookie(default=None)):
    user=optional_user(authorization,eidomira_access_token)
    if user["id"] == "local-guest": raise HTTPException(401,"Authentication required")
    return user
