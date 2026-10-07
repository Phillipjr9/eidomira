"""The engine actually calls the refinement stages.

`app/enhance.py` is tested on its own, and that is not enough: the stages only change the
output if `InSwapperEngine.process` reaches them, with the target frame as the tone
reference and the same mask the composite was built from. A stage that is imported and
never called passes every unit test in `test_enhance.py`.

`InSwapperEngine.__init__` needs insightface, which is not installed here, so the object
is built directly and given stand-ins. What is under test is the wiring, not the models.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.engines.inswapper import InSwapperEngine

H, W = 200, 260
BOX = (80, 50, 180, 150)
#: How far the stand-in swapper's face sits from the room it is pasted into.
MISMATCH = 60


def room() -> np.ndarray:
    """A frame the swap is applied to, with texture so a seam would be visible."""
    texture = np.zeros((H, W, 3), np.int16)
    texture[::3, ::3] = 12
    return np.clip(168 + texture, 0, 255).astype(np.uint8)


class Face:
    def __init__(self, bbox=BOX):
        self.bbox = bbox


class Analyzer:
    """Returns one face, or none, mimicking the InsightFace detector's interface."""

    def __init__(self, found=True):
        self.found = found
        self.calls = 0

    def get(self, _bgr):
        self.calls += 1
        return [Face()] if self.found else []


class Swapper:
    """Reproduces the only thing that matters here: a face from a darker room."""

    def __init__(self, mismatch=MISMATCH):
        self.mismatch = mismatch
        self.calls = 0

    def get(self, bgr, _target, _identity, paste_back=True):
        self.calls += 1
        out = bgr.copy()
        x1, y1, x2, y2 = BOX
        region = out[y1:y2, x1:x2].astype(int) - self.mismatch
        out[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
        return out


class StubRestorer:
    def __init__(self):
        self.calls = []

    def enhance(self, frame_rgb, bbox):
        self.calls.append(tuple(bbox))
        return frame_rgb


def build_engine(*, compositor=None, restorer=None, found=True):
    """An InSwapperEngine with the models replaced. No insightface, no onnxruntime."""
    engine = InSwapperEngine.__new__(InSwapperEngine)
    engine.threshold = .34
    engine.providers = ["CPUExecutionProvider"]
    engine.provider = "cpu"
    engine.accelerated = False
    engine.analyzer = Analyzer(found=found)
    engine.swapper = Swapper()
    engine.compositor = compositor
    engine.restorer = restorer
    return engine


def face_region(frame: np.ndarray) -> float:
    x1, y1, x2, y2 = BOX
    return float(frame[y1:y2, x1:x2].mean())


def surround(frame: np.ndarray) -> float:
    band = frame[10:40, 10:70]
    return float(band.mean())


def test_the_engine_pulls_the_swapped_face_into_the_targets_lighting():
    """The end-to-end claim, measured with the project's own quality tool.

    Note the residual: the stand-in swapper darkens a rectangle *smaller* than the mask,
    so the mask interior still holds a ring of unchanged target skin, which drags the
    statistics and leaves a gap of about 26 levels in place of 60. A real swapper replaces
    the whole face the mask is derived from, so this construction understates the effect
    rather than flattering it.
    """
    from tools.quality_report import colour_shift

    frame = room()
    result = build_engine().process(frame, identity=None, verified=True)

    assert result.face_found
    assert colour_shift(frame, result.image)["distance"] < 10
    gap = surround(result.image) - face_region(result.image)
    assert gap < MISMATCH / 2, f"only closed to {gap:.1f} of {MISMATCH} levels"


def test_without_the_stage_the_mismatch_is_still_there(monkeypatch):
    """The control, so the assertion above cannot pass for an unrelated reason."""
    from tools.quality_report import colour_shift

    monkeypatch.setattr(settings, "tone_transfer_strength", 0.0)
    frame = room()
    result = build_engine().process(frame, identity=None, verified=True)

    assert colour_shift(frame, result.image)["distance"] > 80
    gap = surround(result.image) - face_region(result.image)
    assert gap > MISMATCH * 0.7, f"expected the {MISMATCH}-level gap to remain, saw {gap:.1f}"


def test_the_engine_tolerates_a_disabled_stage(monkeypatch):
    monkeypatch.setattr(settings, "tone_transfer_strength", 0.0)
    engine = build_engine()
    assert engine.process(room(), identity=None, verified=True).face_found


def test_the_compositors_mask_is_the_one_used_for_tone_transfer(monkeypatch):
    """The correction has to be faded with the same mask the face was blended with.

    A rectangular mask would recolour the hair and background inside the box; the
    compositor's mask is the parsed face region only.
    """
    seen = {}

    import app.engines.inswapper as module

    original = module.transfer_tone

    def capture(reference, source, mask, strength):
        seen["mask"] = mask
        return original(reference, source, mask, strength)

    monkeypatch.setattr(module, "transfer_tone", capture)

    class Compositor:
        def __init__(self):
            self.calls = 0

        def blend(self, original_rgb, swapped_rgb, bbox):
            self.calls += 1
            mask = np.zeros(original_rgb.shape[:2], np.float32)
            mask[60:140, 90:170] = 1.0
            out = original_rgb.copy()
            out[60:140, 90:170] = swapped_rgb[60:140, 90:170]
            return out, mask

    compositor = Compositor()
    build_engine(compositor=compositor).process(room(), identity=None, verified=True)

    assert compositor.calls == 1
    mask = seen["mask"]
    assert mask.shape == (H, W)
    # the compositor's mask, not the face box
    assert mask[100, 160] > 0.5
    assert mask[100, 75] == 0.0, "tone transfer used the bbox instead of the parsed mask"


def test_the_restorer_runs_before_the_composite_and_gets_the_face_box():
    restorer = StubRestorer()
    build_engine(restorer=restorer).process(room(), identity=None, verified=True)
    assert restorer.calls == [BOX]


def test_the_restorer_is_skipped_when_no_model_is_configured():
    """No file, no stage — and no crash."""
    engine = build_engine(restorer=None)
    assert engine.process(room(), identity=None, verified=True).face_found


def test_no_face_means_no_refinement_and_the_frame_passes_through():
    engine = build_engine(found=False)
    frame = room()
    result = engine.process(frame, identity=None, verified=True)

    assert not result.face_found
    assert (result.image == frame).all(), "a frame with no face must not be modified"
    assert engine.swapper.calls == 0, "the swapper ran without a detected face"


def test_refinement_softens_the_edge_it_did_not_create(monkeypatch):
    """Matching the average colour should also reduce the step at the boundary, not
    trade one artifact for another."""
    from tools.quality_report import seam_ratio

    frame = room()

    monkeypatch.setattr(settings, "tone_transfer_strength", 0.0)
    before = seam_ratio(frame, build_engine().process(frame, identity=None, verified=True).image)

    monkeypatch.setattr(settings, "tone_transfer_strength", 1.0)
    after = seam_ratio(frame, build_engine().process(frame, identity=None, verified=True).image)

    assert before is not None and after is not None
    assert after < before, f"seam went from {before:.2f} to {after:.2f}"

    # The label is drawn into both frames and has a hard edge of its own, so this is the
    # floor no swap can go below: measured 2.1 with the stage off and no mismatch at all.
    assert after < 4.0


def test_the_label_is_drawn_after_refinement_not_underneath_it(monkeypatch):
    """The "SYNTHETIC" label is a safety boundary. Drawing it last is what keeps it
    legible: run tone transfer over it and the text is recoloured along with the face,
    and past the bottom of the mask it is blended away entirely."""
    frame = room()
    drawn = build_engine().process(frame, identity=None, verified=True).image

    monkeypatch.setattr(settings, "tone_transfer_strength", 0.0)
    without = build_engine().process(frame, identity=None, verified=True).image

    label = (slice(H - 20, H), slice(5, 70))
    assert (drawn[label] != frame[label]).any(), "no label was drawn"
    assert (drawn[label] == without[label]).all(), (
        "refinement changed the label region, so the label is not drawn last"
    )
