import hashlib
import hmac
from app.config import settings
from app.paystack import valid_signature


def test_paystack_signature(monkeypatch):
    monkeypatch.setattr(settings,"paystack_secret_key","sk_test_example")
    body=b'{"event":"charge.success","data":{"reference":"eid_test"}}'
    signature=hmac.new(b"sk_test_example",body,hashlib.sha512).hexdigest()
    assert valid_signature(body,signature)
    assert not valid_signature(body,"bad")
