from app.limits import SlidingWindowLimiter


def test_limiter_blocks_after_limit():
    limiter=SlidingWindowLimiter()
    assert limiter.allow("x",2,60)[0]
    assert limiter.allow("x",2,60)[0]
    allowed,retry=limiter.allow("x",2,60)
    assert not allowed and retry>0


def test_windows_are_per_key():
    """One caller exhausting a limit must not affect another."""
    limiter = SlidingWindowLimiter()
    assert limiter.allow("a", 1, 60)[0]
    assert not limiter.allow("a", 1, 60)[0]
    assert limiter.allow("b", 1, 60)[0]


def test_expired_events_leave_the_window():
    limiter = SlidingWindowLimiter()
    assert limiter.allow("x", 1, 0)[0]
    assert limiter.allow("x", 1, 0)[0]  # zero-length window: previous entry already stale


def test_record_false_checks_without_spending_budget():
    """The property the sign-in throttle relies on: asking must not consume."""
    limiter = SlidingWindowLimiter()
    for _ in range(50):
        allowed, retry = limiter.allow("x", 3, 60, record=False)
        assert allowed and retry == 0
    assert limiter.allow("x", 3, 60)[0]  # budget untouched


def test_record_false_still_reports_an_exhausted_bucket():
    limiter = SlidingWindowLimiter()
    for _ in range(3):
        limiter.allow("x", 3, 60)
    allowed, retry = limiter.allow("x", 3, 60, record=False)
    assert not allowed and retry > 0


def test_retry_after_is_positive_seconds():
    limiter = SlidingWindowLimiter()
    limiter.allow("x", 1, 90)
    _, retry = limiter.allow("x", 1, 90)
    assert 0 < retry <= 90
