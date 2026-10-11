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


def test_blending_converges_exactly_onto_the_true_value():
    """Truncating the cast left the region one level short of the truth, permanently.

    The blend feeds its own output forward, so an error of one level never decays: it
    approaches the true value asymptotically and then sits one level below it for good.
    """
    background = np.full((48, 48, 3), 120, np.uint8)
    swapped = background.copy()
    swapped[16:32, 16:32] = 30

    stabilizer = MotionAwareStabilizer(strength=.22, motion_threshold=24.0)
    for _ in range(10):
        stabilizer.apply(background, swapped)

    for frame in range(6):
        out = stabilizer.apply(background, background)
        if (out == background).all():
            break

    residual = int(np.abs(out.astype(int) - background.astype(int)).max())
    assert residual == 0, f"still {residual} level(s) off after {frame + 1} frames"


def test_reset_restores_pass_through():
    """What the pipeline relies on when a frame produced no swap at all."""
    stabilizer = MotionAwareStabilizer(strength=.5)
    plain = np.full((16, 16, 3), 10, np.uint8)     # input is static, so history dominates
    bright = np.full((16, 16, 3), 200, np.uint8)
    dark = np.zeros((16, 16, 3), np.uint8)

    stabilizer.apply(plain, bright)                 # establishes the previous frame
    blended = stabilizer.apply(plain, dark)
    assert int(blended.mean()) == 100, f"expected a half-way blend, got {int(blended.mean())}"

    stabilizer.reset()
    assert (stabilizer.apply(plain, dark) == dark).all(), "reset did not clear history"


def test_output_stays_within_range():
    stabilizer = MotionAwareStabilizer(strength=.65)
    dark = np.zeros((16, 16, 3), np.uint8)
    bright = np.full((16, 16, 3), 255, np.uint8)
    stabilizer.apply(dark, bright)
    for _ in range(5):
        out = stabilizer.apply(bright, dark)
        assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255
