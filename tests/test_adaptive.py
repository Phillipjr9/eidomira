from app.adaptive import AdaptiveQualityController


def test_reduces_width_when_slow():
    q = AdaptiveQualityController("balanced", 960, 384, 8)
    start = q.width
    for _ in range(8):
        q.observe(100)
    assert q.width < start


def test_never_exceeds_preset_maximum():
    q = AdaptiveQualityController("speed", 1920, 384, 8)
    for _ in range(32):
        q.observe(1)
    assert q.width <= 512


def test_never_below_minimum():
    q = AdaptiveQualityController("quality", 960, 384, 8)
    for _ in range(100):
        q.observe(500)
    assert q.width >= 384
