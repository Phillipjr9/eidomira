import jwt
from app.calls import clean, create_call_token
from app.config import settings


def test_clean_room_name():
    assert clean(" My unsafe room! ") == "My-unsafe-room"


def test_call_token_has_restricted_room(monkeypatch):
    monkeypatch.setattr(settings, "livekit_api_key", "key")
    monkeypatch.setattr(settings, "livekit_api_secret", "secret")
    result = create_call_token("eidomira-test", "Ada")
    payload = jwt.decode(result["token"], "secret", algorithms=["HS256"], audience=None,
                         options={"verify_aud": False})
    assert payload["video"]["room"] == "eidomira-test"
    assert payload["video"]["roomJoin"] is True
