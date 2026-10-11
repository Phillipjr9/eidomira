"""Post-swap refinement: tone transfer and the restoration stage.

Tone transfer is measured with `tools.quality_report`, so the claim that it improves the
composite is checked by the same metric that would be used to judge a real swap rather
than by a test that only proves the code ran.

Restoration is exercised against a stand-in onnxruntime, because the model file is a
licensing decision and is not in the repository.
"""
from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest

from app.enhance import FaceRestorer, transfer_tone
from app.providers import execution_providers
from tools.quality_report import colour_shift, seam_ratio

H, W = 200, 260
SKIN = (slice(50, 150), slice(80, 180))
BOX = (80, 50, 180, 150)


def scene() -> np.ndarray:
    texture = np.zeros((H, W, 3), np.int16)
    texture[::3, ::3] = 12
    texture[1::5, 2::5] = -6
    return np.clip(168 + texture, 0, 255).astype(np.uint8)


def face_frame(offset: int, own_texture: bool = True) -> np.ndarray:
    """A face with its own structure, sitting `offset` levels away from the scene."""
    texture = np.zeros((H, W, 3), np.int16)
    if own_texture:
        texture[::4, 1::4] = -11
    return np.clip(169 + texture + offset, 0, 255).astype(np.uint8)


def feathered_mask() -> np.ndarray:
    mask = np.zeros((H, W), np.float32)
    mask[SKIN] = 1.0
    return cv2.GaussianBlur(mask, (21, 21), 0)


def composite(incoming: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    original, a = scene(), feathered_mask() if alpha is None else alpha
    a = a[..., None]
    blended = original.astype(np.float32) * (1 - a) + incoming.astype(np.float32) * a
    return np.rint(np.clip(blended, 0, 255)).astype(np.uint8)


# ─────────────────────────────── tone transfer ───────────────────────────────

def test_tone_transfer_reduces_a_lighting_mismatch():
    """The measurement that matters: the tool's colour distance has to improve."""
    original = scene()
    swapped = composite(face_frame(-25))
    before = colour_shift(original, swapped)["distance"]
    after = colour_shift(original, transfer_tone(original, swapped, feathered_mask()))["distance"]
    assert before > 25, before
    assert after < before * 0.4, f"colour distance only went {before} -> {after}"


def test_a_well_lit_face_is_barely_touched():
    """It corrects a mismatch; it must not recolour a face that already matches."""
    original = scene()
    matched = composite(face_frame(0))
    out = transfer_tone(original, matched, feathered_mask())
    change = np.abs(out.astype(int) - matched.astype(int))
    assert change.mean() < 0.5, f"mean change {change.mean():.2f} levels on a matched face"


def test_pixels_the_mask_does_not_cover_are_left_exactly_alone():
    original = scene()
    swapped = composite(face_frame(-25))
    mask = np.zeros((H, W), np.float32)
    mask[100:140, 120:160] = 1.0

    out = transfer_tone(original, swapped, mask)
    untouched = np.ones((H, W), bool)
    untouched[100:140, 120:160] = False
    assert (out[untouched] == swapped[untouched]).all()


def test_strength_is_a_dial():
    original = scene()
    swapped = composite(face_frame(-25))
    mask = feathered_mask()

    off = transfer_tone(original, swapped, mask, strength=0.0)
    assert (off == swapped).all(), "strength 0 must change nothing"

    full = colour_shift(original, transfer_tone(original, swapped, mask, strength=1.0))["distance"]
    half = colour_shift(original, transfer_tone(original, swapped, mask, strength=0.5))["distance"]
    unmatched = colour_shift(original, swapped)["distance"]
    assert full < half < unmatched


def test_grain_is_not_blown_up_to_the_reference_contrast():
    """The clamp's real job. An unbounded gain matches the reference's *contrast*, so a
    low-contrast face carrying sensor grain gets its grain stretched to full texture
    strength — visible noise over the whole face."""
    textured = scene().copy()
    checker = np.zeros((H, W, 3), np.int16)
    checker[::2, :] = 25
    checker[1::2, :] = -25
    textured = np.clip(168 + checker, 0, 255).astype(np.uint8)

    rng = np.random.default_rng(4)
    flat = face_frame(-25)
    flat[SKIN] = np.clip(140 + rng.integers(-3, 4, flat[SKIN].shape), 0, 255).astype(np.uint8)
    swapped = composite(flat)
    mask = feathered_mask()
    interior = cv2.erode((mask > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    def contrast(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        return float(lab[..., 0][interior].std())

    source = contrast(swapped)
    limited = contrast(transfer_tone(textured, swapped, mask, gain_limit=(0.8, 1.25)))
    unlimited = contrast(transfer_tone(textured, swapped, mask, gain_limit=(0.0, 1e6)))

    assert limited / source <= 1.30, f"clamp did not hold: {limited / source:.2f}x"
    assert unlimited / source > 3.0, (
        f"without the clamp contrast grows {unlimited / source:.2f}x, so this case does "
        "not exercise the clamp and the assertion above proves nothing"
    )


def test_tone_transfer_preserves_shape_and_dtype():
    original = scene()
    out = transfer_tone(original, composite(face_frame(-25)), feathered_mask())
    assert out.shape == original.shape and out.dtype == np.uint8


def test_a_mask_too_small_to_measure_is_ignored():
    original = scene()
    swapped = composite(face_frame(-25))
    tiny = np.zeros((H, W), np.float32)
    tiny[10:14, 10:14] = 1.0                      # 16 pixels, below the floor
    assert (transfer_tone(original, swapped, tiny) == swapped).all()


def test_mismatched_shapes_are_refused():
    original = scene()
    with pytest.raises(ValueError):
        transfer_tone(original, original[:100], feathered_mask())
    with pytest.raises(ValueError):
        transfer_tone(original, original, np.zeros((10, 10), np.float32))


def test_tone_transfer_does_not_introduce_a_seam():
    """Matching colour must not trade one artifact for another."""
    original = scene()
    swapped = composite(face_frame(-25))
    before = seam_ratio(original, swapped)
    after = seam_ratio(original, transfer_tone(original, swapped, feathered_mask()))
    assert after is None or before is None or after <= before * 1.2


# ───────────────────────────── the restoration stage ─────────────────────────────

class FakeInput:
    def __init__(self, name="input", shape=(1, 3, 64, 64)):
        self.name, self.shape = name, shape


class FakeSession:
    """Replaces an onnxruntime session with a deterministic restoration."""

    created: list = []

    def __init__(self, output=None, shape=(1, 3, 64, 64), raises=False):
        self.output, self.shape, self.raises = output, shape, raises
        self.calls = 0
        FakeSession.created.append(self)

    def get_inputs(self):
        return [FakeInput(shape=self.shape)]

    def run(self, _outputs, _feeds):
        self.calls += 1
        if self.raises:
            raise RuntimeError("model does not accept this input")
        return [self.output]


@pytest.fixture
def fake_ort(monkeypatch):
    module = types.ModuleType("onnxruntime")

    def factory(model_path, providers=None):
        requested = {"providers": providers, "path": model_path}
        session = FakeSession()
        session.requested = requested
        return session

    module.InferenceSession = factory
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    FakeSession.created = []
    return module


def restorer(fake_ort, output, visibility=.75, size=64) -> FaceRestorer:
    """Build a real FaceRestorer, with the session swapped for one we control."""
    import app.enhance as enhance

    class Session(FakeSession):
        def __init__(self, model_path, providers=None):
            super().__init__(output=output)
            self.requested = {"providers": providers, "path": model_path}

    original = sys.modules["onnxruntime"].InferenceSession

    def factory(model_path, providers=None):
        return Session(model_path, providers)

    sys.modules["onnxruntime"].InferenceSession = factory
    try:
        instance = FaceRestorer("gfpgan_1.4.onnx", visibility=visibility, size=size)
    finally:
        sys.modules["onnxruntime"].InferenceSession = original
    return instance


def test_restoration_uses_the_provider_registry(fake_ort):
    instance = restorer(fake_ort, np.zeros((3, 64, 64), np.uint8))
    assert instance.providers == list(execution_providers())


def test_visibility_zero_skips_the_model_entirely(fake_ort):
    """A restorer switched off must not cost inference."""
    frame = scene()
    instance = restorer(fake_ort, np.zeros((3, 64, 64), np.uint8), visibility=0.0)
    assert (instance.enhance(frame, BOX) == frame).all()


def test_restoration_moves_the_face_towards_the_model_output(fake_ort):
    frame = scene()
    bright = np.full((3, 64, 64), 250, np.uint8)
    out = restorer(fake_ort, bright, visibility=1.0).enhance(frame, BOX)
    x1, y1, x2, y2 = BOX
    assert out[y1:y2, x1:x2].mean() > frame[y1:y2, x1:x2].mean() + 20


def test_visibility_controls_how_much_of_it_reaches_the_frame(fake_ort):
    frame = scene()
    bright = np.full((3, 64, 64), 250, np.uint8)
    full = restorer(fake_ort, bright, visibility=1.0).enhance(frame, BOX)
    part = restorer(fake_ort, bright, visibility=.5).enhance(frame, BOX)
    x1, y1, _, _ = BOX
    assert full[y1 + 20, x1 + 20].mean() > part[y1 + 20, x1 + 20].mean() > frame[y1 + 20, x1 + 20].mean()


def test_pixels_outside_the_box_are_untouched(fake_ort):
    frame = scene()
    out = restorer(fake_ort, np.full((3, 64, 64), 250, np.uint8)).enhance(frame, BOX)
    outside = np.ones((H, W), bool)
    x1, y1, x2, y2 = BOX
    outside[y1:y2, x1:x2] = False
    assert (out[outside] == frame[outside]).all()


def test_a_box_too_small_to_restore_is_skipped(fake_ort):
    frame = scene()
    instance = restorer(fake_ort, np.full((3, 64, 64), 250, np.uint8))
    assert (instance.enhance(frame, (10, 10, 14, 14)) == frame).all()


def test_a_model_that_rejects_the_input_leaves_the_frame_alone(fake_ort):
    """Better a soft face than a broken stream."""
    frame = scene()
    instance = restorer(fake_ort, np.zeros((3, 64, 64), np.uint8))
    instance.session.raises = True
    assert (instance.enhance(frame, BOX) == frame).all()


def test_an_unexpected_model_output_leaves_the_frame_alone(fake_ort):
    frame = scene()
    instance = restorer(fake_ort, np.zeros((5, 5), np.uint8))     # not a face image
    assert (instance.enhance(frame, BOX) == frame).all()


def test_a_failing_model_is_reported_rather_than_silently_ignored(fake_ort, caplog):
    """A model that fails every frame must not look identical to a working one."""
    frame = scene()
    instance = restorer(fake_ort, np.zeros((3, 64, 64), np.uint8))
    instance.session.raises = True

    with caplog.at_level("WARNING", logger="eidomira.enhance"):
        assert (instance.enhance(frame, BOX) == frame).all()

    assert "model does not accept this input" in caplog.text
    assert instance.fault is not None


def test_the_fault_is_reported_once_not_every_frame(fake_ort, caplog):
    frame = scene()
    instance = restorer(fake_ort, np.zeros((3, 64, 64), np.uint8))
    instance.session.raises = True

    with caplog.at_level("WARNING", logger="eidomira.enhance"):
        for _ in range(5):
            instance.enhance(frame, BOX)

    assert len(caplog.records) == 1, f"logged {len(caplog.records)} times for 5 frames"


def test_an_unexpected_output_shape_is_reported(fake_ort, caplog):
    frame = scene()
    instance = restorer(fake_ort, np.zeros((5, 5), np.uint8))

    with caplog.at_level("WARNING", logger="eidomira.enhance"):
        instance.enhance(frame, BOX)

    assert "unexpected shape" in caplog.text, caplog.text


def test_a_clean_session_records_no_fault(fake_ort):
    frame = scene()
    instance = restorer(fake_ort, np.full((3, 64, 64), 200, np.uint8))
    instance.enhance(frame, BOX)
    assert instance.fault is None


def test_float_model_output_is_scaled(fake_ort):
    frame = scene()
    half = np.full((1, 3, 64, 64), 0.98, np.float32)              # 0..1 range
    out = restorer(fake_ort, half, visibility=1.0).enhance(frame, BOX)
    x1, y1, _, _ = BOX
    assert out[y1 + 20, x1 + 20].mean() > 200
