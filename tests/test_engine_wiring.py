"""The engine actually calls the refinement stages.

`app/enhance.py` is tested on its own, and that is not enough: the stages only change the
output if `InSwapperEngine.process` reaches them, with the target frame as the tone
reference and the same mask the composite was built from. A stage that is imported and
never called passes every unit test in `test_enhance.py`.

`InSwapperEngine.__init__` needs insightface, which is not installed here, so the object
is built directly and given stand-ins. What is under test is the wiring, not the models.
"""
from __future__ import annotations

import cv2
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
        self.fault = None

    def enhance(self, frame_rgb, bbox, visibility=None):
        self.calls.append((tuple(bbox), visibility))
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
            self.fault = None

        def blend(self, original_rgb, swapped_rgb, bbox, feather=None):
            self.calls += 1
            self.feather = feather
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
    assert [call[0] for call in restorer.calls] == [BOX]
    # no override, so the configured default is what reached the stage
    assert restorer.calls[0][1] == settings.restoration_visibility


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


# ───────────────────────── the sub-pixel boost, wired in ─────────────────────────

#: The boost tests need a face *larger than the 128 crop*, because that is the only case
#: where the phases recover anything: the crop has to have thrown detail away first. The
#: shared BOX above is 100 pixels across, which the crop enlarges, so these tests bring
#: their own geometry rather than distorting the ones the rest of the file is built on.
BIG_BOX = (80, 50, 280, 250)                 # 200 pixels across
BIG_H, BIG_W = 320, 400
CROP_TO_BIG = np.array([[.64, 0, -BIG_BOX[0] * .64], [0, .64, -BIG_BOX[1] * .64]], np.float64)


def big_room() -> np.ndarray:
    """The shared room's texture on a frame with space for a 200-pixel face."""
    texture = np.zeros((BIG_H, BIG_W, 3), np.int16)
    texture[::3, ::3] = 12
    return np.clip(168 + texture, 0, 255).astype(np.uint8)


def big_face(frame: np.ndarray) -> float:
    x1, y1, x2, y2 = BIG_BOX
    return float(frame[y1:y2, x1:x2].mean())


def big_surround(frame: np.ndarray) -> float:
    return float(frame[5:35, 5:65].mean())


@pytest.fixture
def fake_insightface(monkeypatch):
    """The two things `_boosted_swap` takes from insightface, and nothing else.

    Standing the library up rather than skipping these tests is deliberate: the parts of
    this path that *can* be wrong on this machine — how many passes are made, whether the
    crops differ, whether paste-back lands on the face, whether a small face is skipped —
    are the parts a stand-in exercises. The three assumptions about the real library are
    listed on the method, and no amount of stubbing checks those.
    """
    import sys
    import types

    face_align = types.ModuleType("insightface.utils.face_align")
    face_align.estimate_norm = lambda kps, size, mode="arcface": CROP_TO_BIG.copy()
    face_align.arcface_dst = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]], np.float32,
    )
    utils = types.ModuleType("insightface.utils")
    utils.face_align = face_align
    top = types.ModuleType("insightface")
    top.utils = utils
    monkeypatch.setitem(sys.modules, "insightface", top)
    monkeypatch.setitem(sys.modules, "insightface.utils", utils)
    monkeypatch.setitem(sys.modules, "insightface.utils.face_align", face_align)
    return face_align


class KeypointFace(Face):
    """The engine reads `kps` to align; the stand-in ignores their values."""

    def __init__(self, bbox=BIG_BOX):
        super().__init__(bbox)
        self.kps = np.zeros((5, 2), np.float32)


class AlignedSwapper:
    """A swapper that only understands the aligned path, the way `paste_back=False` does."""

    def __init__(self, mismatch=MISMATCH, blows_up=False):
        self.mismatch = mismatch
        self.blows_up = blows_up
        self.crops = []
        self.paste_back_calls = 0
        self.aligned_calls = 0

    def get(self, bgr, target, identity, paste_back=True):
        if paste_back:
            self.paste_back_calls += 1
            out = bgr.copy()
            x1, y1, x2, y2 = BIG_BOX
            out[y1:y2, x1:x2] = np.clip(
                out[y1:y2, x1:x2].astype(int) - self.mismatch, 0, 255
            ).astype(np.uint8)
            return out
        if self.blows_up:
            raise RuntimeError("onnxruntime said no")
        self.aligned_calls += 1
        self.crops.append(bgr.copy())
        assert bgr.shape[:2] == (128, 128), f"the model gets a 128 crop, got {bgr.shape}"
        return np.clip(bgr.astype(int) - self.mismatch, 0, 255).astype(np.uint8)


def build_boosted_engine(swapper, box=BIG_BOX) -> InSwapperEngine:
    engine = build_engine()
    engine.swapper = swapper
    engine.analyzer = Analyzer()
    engine.analyzer.get = lambda _bgr: [KeypointFace(box)]
    engine._boost_disabled = False
    return engine


def test_boosting_runs_a_pass_per_phase_and_pastes_the_result_onto_the_face(
        monkeypatch, fake_insightface):
    monkeypatch.setattr(settings, "swap_pixel_boost", 2)
    swapper = AlignedSwapper()
    engine = build_boosted_engine(swapper)

    result = engine.process(big_room(), identity=object(), verified=True)

    assert swapper.aligned_calls == 4, "a 2x boost is a 2x2 grid of passes"
    assert swapper.paste_back_calls == 0, (
        "the boost pastes back through app/boost.paste_back; letting the library paste too "
        "would put the unboosted 128 face on top of the boosted one"
    )
    # Four different crops: if the phase were dropped, all four would be the same image and
    # the merge would be a very expensive way to do nothing.
    assert len({crop.tobytes() for crop in swapper.crops}) == 4
    assert big_face(result.image) < big_surround(result.image) - 10
    assert result.face_found is True


def test_a_single_pass_still_goes_through_the_library(monkeypatch, fake_insightface):
    """The default must be the untouched old path, including its own paste-back."""
    monkeypatch.setattr(settings, "swap_pixel_boost", 1)
    swapper = AlignedSwapper()
    engine = build_boosted_engine(swapper)

    engine.process(big_room(), identity=object(), verified=True)

    assert swapper.paste_back_calls == 1
    assert swapper.aligned_calls == 0


def test_a_boost_that_fails_falls_back_and_stops_trying(monkeypatch, fake_insightface, caplog):
    """A library that does not match the assumptions must cost one frame, not every frame."""
    monkeypatch.setattr(settings, "swap_pixel_boost", 2)
    swapper = AlignedSwapper(blows_up=True)
    engine = build_boosted_engine(swapper)

    with caplog.at_level("WARNING"):
        first = engine.process(big_room(), identity=object(), verified=True)
        second = engine.process(big_room(), identity=object(), verified=True)

    assert first.face_found and second.face_found, "the swap should still happen"
    assert swapper.paste_back_calls == 2, "both frames should have taken the single pass"
    warnings_sent = [r for r in caplog.records if "pixel boost failed" in r.getMessage()]
    assert len(warnings_sent) == 1, (
        "a deployment failing every boosted frame must not log every frame: "
        f"{len(warnings_sent)} warnings for two frames"
    )
    assert engine._boost_disabled is True


def test_the_boost_leaves_the_rest_of_the_frame_alone(monkeypatch, fake_insightface):
    """The canvas covers a region and the layer is zero outside it. Compositing that layer
    without its alpha turns the rest of the frame black, which is what this caught."""
    monkeypatch.setattr(settings, "swap_pixel_boost", 2)
    engine = build_boosted_engine(AlignedSwapper())

    result = engine.process(big_room(), identity=object(), verified=True)

    assert np.array_equal(result.image[5, 5], big_room()[5, 5]), "the corner is room"
    assert np.array_equal(result.image[BIG_H - 5, BIG_W - 5],
                          big_room()[BIG_H - 5, BIG_W - 5])


def test_a_face_too_small_to_have_lost_detail_is_not_boosted(monkeypatch, fake_insightface):
    """A crop scale at or above 1 means the crop enlarged a face that was already small, so
    nothing the phases recover was thrown away and the passes would cost four times for it."""
    small_box = (40, 30, 120, 110)          # 80 px, so the 128 crop is an upscale
    monkeypatch.setattr(settings, "swap_pixel_boost", 2)
    fake_insightface.estimate_norm = lambda kps, size, mode="arcface": np.array(
        [[128 / 80, 0, -small_box[0] * 128 / 80], [0, 128 / 80, -small_box[1] * 128 / 80]]
    )
    swapper = AlignedSwapper()
    engine = build_boosted_engine(swapper, box=small_box)

    result = engine.process(room(), identity=object(), verified=True)

    assert result.face_found
    assert swapper.aligned_calls == 0, "a small face should not pay for phases"
    assert swapper.paste_back_calls == 1
    assert engine._boost_disabled is False, "skipping is not a failure"


def test_boosting_sharpens_a_face_the_single_pass_had_to_downsample(
        monkeypatch, fake_insightface):
    """The end-to-end reason the boost exists, through the engine, measured.

    A face 200 pixels across does not fit in a 128 crop, so one pass samples it at 0.64x and
    whatever was finer than that is gone: pasting the 128 result back cannot invent it. The
    boost samples the same face at 1.28x across a 256 canvas, so the detail survives the
    round trip and lands in the frame — measured against the face as it really is.
    """
    frame_bgr = big_room()
    face = np.full((200, 200, 3), 170, np.uint8)
    checkerboard = np.indices((200, 200)).sum(0) % 2
    face[checkerboard == 1] = np.array([130, 120, 210], np.uint8)
    frame_bgr[BIG_BOX[1]:BIG_BOX[3], BIG_BOX[0]:BIG_BOX[2]] = face
    true_face = cv2.cvtColor(frame_bgr[BIG_BOX[1]:BIG_BOX[3], BIG_BOX[0]:BIG_BOX[2]],
                             cv2.COLOR_BGR2GRAY)

    class LibrarySwapper:
        """Both paths, with the resolution limit a 128 model really has."""

        def get(self, bgr, target, identity, paste_back=True):
            if paste_back:
                # What the library does: align to 128, run, warp the 128 result back.
                crop = cv2.warpAffine(bgr, CROP_TO_BIG, (128, 128), flags=cv2.INTER_AREA,
                                      borderMode=cv2.BORDER_REPLICATE)
                inverse = cv2.invertAffineTransform(CROP_TO_BIG)
                return cv2.warpAffine(crop, inverse, (bgr.shape[1], bgr.shape[0]),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return bgr.copy()

    def run(boost):
        monkeypatch.setattr(settings, "swap_pixel_boost", boost)
        engine = build_boosted_engine(LibrarySwapper())
        return np.array(engine.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                                       object(), True).image)

    def face_grey(rgb):
        return cv2.cvtColor(np.ascontiguousarray(rgb[BIG_BOX[1]:BIG_BOX[3],
                                                     BIG_BOX[0]:BIG_BOX[2]]),
                            cv2.COLOR_RGB2GRAY)

    detail = lambda g: float(cv2.Laplacian(g, cv2.CV_64F).var())
    single_detail = detail(face_grey(run(1)))
    boosted_detail = detail(face_grey(run(2)))
    true_detail = detail(true_face)

    assert true_detail > 1000, f"the test face should be detailed, measured {true_detail:.0f}"
    assert single_detail < boosted_detail, (
        f"one pass kept {single_detail:.0f} of {true_detail:.0f}; "
        f"the boost kept {boosted_detail:.0f}"
    )
    assert boosted_detail > single_detail * 1.5, (
        "the boost should recover markedly more of the face's detail, not a few percent: "
        f"{single_detail:.0f} then {boosted_detail:.0f} against {true_detail:.0f}"
    )
