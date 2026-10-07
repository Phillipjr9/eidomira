from app import ice
from app.config import settings


def test_stun_is_always_present():
    config = ice.rtc_configuration("session")
    assert config["iceServers"][0]["urls"][0].startswith("stun:")


def test_turn_rest_credentials(monkeypatch):
    monkeypatch.setattr(settings, "turn_urls", "turn:example.com:3478")
    monkeypatch.setattr(settings, "turn_secret", "test-secret")
    config = ice.rtc_configuration("abc")
    turn = config["iceServers"][1]
    assert turn["username"].endswith(":abc")
    assert turn["credential"]
