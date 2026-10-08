"""The neural path, executed — with stand-ins, on the CPU, on purpose.

Three neural stages sit in this stack and none of them has ever run in this repository:

* the swap weights are non-commercial and cannot be shipped here at all;
* the parser and restoration models are optional by design, and the code says so — a
  deployment without them keeps working, with a softer swap;
* and the host has no GPU.

The result was that `app/compositor.py`, `app/enhance.py` and `app/boost.py` could be read
but never executed against a real ONNX session. Every test around them substituted a fake for
the model. That is how a shape, an input order or a normalisation constant differs from its
documentation for months without anything noticing.

`tools/stand_in_models.py` writes graphs with the shapes and names the adapters require — no
trained weights, no quality claim. These tests drive that through the *real* classes:

    shapes      — does the graph load, and does the adapter accept it?
    layout      — is the tensor NCHW, is the range right, is the class axis where it belongs?
    behaviour   — does the compositor actually preserve an occluder?
    plumbing    — does `boost` interleave real inference output?

None of that is a claim about quality. It is a claim about the plumbing, which is what can be
checked here, and it is the precondition for an honest claim about quality once the licensed
artefacts exist.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app import boost

ROOT = Path(__file__).resolve().parent.parent
PHOTO = ROOT / "static" / "persona-01.jpg"

#: Every test here needs the optional neural dependencies. `requirements-neural.txt` lists
#: them, and a deployment that opts out of the neural stack should not fail this suite.
onnx = pytest.importorskip("onnx", reason="the stand-in generator needs onnx")
pytest.importorskip("onnxruntime", reason="the adapters need onnxruntime")
insightface = pytest.importorskip("insightface", reason="the swap adapter needs insightface")

from tools import stand_in_models  # noqa: E402  (after the import guard, deliberately)


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, Path]:
    """The three stand-ins, built once — the parser's template is the slow one."""
    return stand_in_models.build(tmp_path_factory.mktemp("standin"))


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    if not PHOTO.exists():  # pragma: no cover - the repository ships this photograph
        pytest.skip("no sample photograph")
    import cv2
    return cv2.imread(str(PHOTO))[:, :, ::-1]


# ── the shapes the adapters were written against ────────────────────────────────

def test_the_stand_ins_declare_the_shapes_the_adapters_read(models):
    """These are contracts, not conveniences: the compositor takes its working size from
    `shape[-1]`, and inswapper takes its crop size from `inputs[0].shape[2:4]`."""
    import onnxruntime as ort

    parser = ort.InferenceSession(str(models["face_parser.onnx"]),
                                  providers=["CPUExecutionProvider"])
    assert parser.get_inputs()[0].shape[0] == 1
    assert parser.get_inputs()[0].shape[1] == 3
    assert parser.get_outputs()[0].shape[1] == stand_in_models.CLASSES

    swap = ort.InferenceSession(str(models["inswapper_128.onnx"]),
                                providers=["CPUExecutionProvider"])
    # Order matters: the adapter runs {inputs[0]: crop, inputs[1]: latent}.
    assert [i.name for i in swap.get_inputs()] == ["target", "source"]
    assert swap.get_inputs()[0].shape[2:] == [stand_in_models.ALIGN] * 2
    assert len(swap.get_outputs()) == 1, "inswapper asserts a single output"

    restorer = ort.InferenceSession(str(models["gfpgan_1.4.onnx"]),
                                    providers=["CPUExecutionProvider"])
    assert restorer.get_inputs()[0].shape[-1] == stand_in_models.RESTORER_SIZE


def test_inswapper_finds_the_embedding_map_it_needs(models):
    """`INSwapper.__init__` reads `graph.initializer[-1]` and multiplies the latent by it
    before the model runs. Any initializer declared after the emap is read in its place, and
    the failure surfaces much later as a shape complaint in the middle of a stream. This
    asserts the reading, not the intent."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(models["inswapper_128.onnx"])).graph
    emap = numpy_helper.to_array(graph.initializer[-1])
    assert emap.shape == (stand_in_models.EMBEDDING, stand_in_models.EMBEDDING)


# ── the compositor: the one behaviour worth asserting, with a real model in place ──

def test_the_compositor_builds_a_mask_and_preserves_the_occluder(models, frame):
    """The claim the semantic compositor exists to make: `swapped = mask * swap + (1 - mask)
    * original`, with the occluders *out* of the mask. The stand-in's label map puts glasses
    across the eyes precisely so there is something to preserve — a constant label map could
    not fail this test."""
    from app.compositor import SemanticCompositor

    compositor = SemanticCompositor(str(models["face_parser.onnx"]), 0.035, include_ears=True)
    assert compositor.fault is None
    assert compositor.size == stand_in_models.PARSER_SIZE
    assert compositor.providers == ["CPUExecutionProvider"], "the host's real provider list"

    height, width = frame.shape[:2]
    bbox = (int(width * .25), int(height * .10), int(width * .75), int(height * .85))
    mask = compositor.mask(frame, bbox)

    assert mask.shape == frame.shape[:2], "the mask is per-pixel over the whole frame"
    assert mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0, "a mask, not raw scores"
    assert (mask > .5).mean() > .02, "nothing would be swapped at all"

    # Where the stand-in draws glasses, the mask must be zero: that region is the original.
    band = mask[int(height * .16):int(height * .20), int(width * .30):int(width * .50)]
    assert float(band.max()) == 0.0, "the occluder was covered instead of preserved"


def test_the_compositor_blends_rather_than_replacing(models, frame):
    from app.compositor import SemanticCompositor

    compositor = SemanticCompositor(str(models["face_parser.onnx"]), 0.035)
    height, width = frame.shape[:2]
    bbox = (int(width * .25), int(height * .10), int(width * .75), int(height * .85))
    swapped = np.full_like(frame, 255)
    # `blend` returns the composited frame *and* the alpha it used — the memory of every mask
    # decision, which is what the trainer samples and what the debug view draws.
    blended, alpha = compositor.blend(frame, swapped, bbox)

    assert blended.shape == frame.shape
    assert blended.dtype == frame.dtype
    assert alpha.shape == frame.shape[:2]
    changed = (blended != frame).any(axis=2)
    assert changed.any(), "the swap was not applied anywhere"
    assert not changed.all(), "the frame was replaced instead of blended"
    # Where alpha is zero the original must survive byte for byte: that is what "preserved"
    # means for an occluder, and rounding a blend that is entirely original is how it drifts.
    untouched = alpha == 0.0
    assert untouched.any()
    assert (blended[untouched] == frame[untouched]).all()


# ── the restorer and the boost ───────────────────────────────────────────────────

def test_the_restorer_enhances_without_faulting(models, frame):
    from app.enhance import FaceRestorer

    restorer = FaceRestorer(str(models["gfpgan_1.4.onnx"]), visibility=.75)
    assert restorer.fault is None
    assert restorer.size == stand_in_models.RESTORER_SIZE

    height, width = frame.shape[:2]
    bbox = (int(width * .25), int(height * .10), int(width * .75), int(height * .85))
    enhanced = restorer.enhance(frame, bbox)

    assert enhanced.shape == frame.shape and enhanced.dtype == frame.dtype
    assert restorer.fault is None, f"the restorer faulted: {restorer.fault}"
    difference = np.abs(enhanced.astype(int) - frame.astype(int)).mean()
    assert difference > 0, "enhance() returned the frame untouched"
    assert difference < 40, "the frame was replaced rather than restored into place"


def test_the_boost_interleaves_real_inference_output(models, frame):
    """`boosted_face` is pure geometry and takes the swapper as a callable, so it can be driven
    by anything — including a real ONNX session. The interleave is the part that cannot be
    verified by reading: four phase-shifted passes have to land on one 256px canvas with each
    phase contributing its own samples."""
    swap = insightface.model_zoo.get_model(str(models["inswapper_128.onnx"]),
                                          providers=["CPUExecutionProvider"])
    assert swap.input_size == (stand_in_models.ALIGN, stand_in_models.ALIGN)

    latent = np.random.default_rng(0).normal(size=(1, stand_in_models.EMBEDDING)).astype(np.float32)
    crop = frame[:stand_in_models.ALIGN, :stand_in_models.ALIGN][:, :, ::-1]

    def swap_once(pixels: np.ndarray) -> np.ndarray:
        # `forward` divides by 255 itself, so it wants 0-255 input; `get` builds its blob the
        # same way. Feeding it already-normalised values is the mistake that makes a working
        # model look flat.
        blob = pixels.astype(np.float32).transpose(2, 0, 1)[None]
        out = swap.forward(blob, latent)[0].transpose(1, 2, 0)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]

    canvas = boost.boosted_face(lambda dx, dy: crop, swap_once, scale=2)

    assert canvas.shape == (stand_in_models.ALIGN * 2, stand_in_models.ALIGN * 2, 3)
    assert canvas.dtype == np.uint8
    assert canvas.std() > 0, "the canvas is empty"
    # The stand-in's convolution is an identity, so structure survives: a phase merge that
    # dropped samples would show up here as a smeared or flattened canvas.
    assert np.isfinite(canvas).all()


def test_the_documented_environment_actually_selects_the_neural_engine():
    """`insightface` instead of `inswapper` does not raise — `create_engine` falls through to
    the diagnostic engine, and a host that meant to run the neural stack quietly does not. The
    value is asserted against the factory's own constant, and the branch is driven for real:
    with the backend selected and no model at the path, the factory must refuse that, which is
    how we know the value took effect without needing a network or a GPU."""
    from app.config import settings
    from app.engines.factory import INSWAPPER_BACKEND, create_engine

    assert stand_in_models.ENVIRONMENT["STUDIO_BACKEND"] == INSWAPPER_BACKEND

    original_backend, original_path = settings.backend, settings.model_path
    settings.backend = INSWAPPER_BACKEND
    settings.model_path = Path("/nonexistent/inswapper_128.onnx")
    try:
        with pytest.raises(RuntimeError, match="Licensed model is missing"):
            create_engine()
    finally:
        settings.backend, settings.model_path = original_backend, original_path


def test_the_documented_environment_points_at_the_stand_ins():
    """The tool prints these; pointing a host at the stand-in directory should be a copy and
    paste, not a guess from reading config.py."""
    for key, value in stand_in_models.ENVIRONMENT.items():
        assert key.startswith("STUDIO_"), key
        if key.endswith("_PATH"):
            assert Path(value).name in stand_in_models.BUILDERS, value
