from app.limits import SlidingWindowLimiter


def test_limiter_blocks_after_limit():
    limiter=SlidingWindowLimiter()
    assert limiter.allow("x",2,60)[0]
    assert limiter.allow("x",2,60)[0]
    allowed,retry=limiter.allow("x",2,60)
    assert not allowed and retry>0
