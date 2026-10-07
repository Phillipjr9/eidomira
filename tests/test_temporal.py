import numpy as np
from app.temporal import MotionAwareStabilizer


def test_static_frame_is_smoothed():
    s = MotionAwareStabilizer(strength=.25)
    inp = np.zeros((32, 32, 3), np.uint8)
    first = np.zeros_like(inp)
    second = np.full_like(inp, 100)
    s.apply(inp, first)
    result = s.apply(inp, second)
    assert 70 <= int(result.mean()) <= 80


def test_reset_accepts_new_shape():
    s = MotionAwareStabilizer()
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.zeros((20, 20, 3), np.uint8)
    assert s.apply(a, a).shape == a.shape
    assert s.apply(b, b).shape == b.shape
