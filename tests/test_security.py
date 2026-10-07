from app.security import access_token
import jwt
from app.config import settings


def test_access_token_subject():
    token=access_token({"id":"user1","email":"a@example.com"})
    payload=jwt.decode(token,settings.auth_secret,algorithms=["HS256"])
    assert payload["sub"]=="user1" and payload["typ"]=="access"
