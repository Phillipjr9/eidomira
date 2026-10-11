"""Seam blending: which pixels the swapped face is allowed to cover.

`test_class_sets_protect_glasses_and_hair` checked that the class sets *say* the right
thing. These check that the code actually applies them — the constant was declared and
never read, so the morphological close painted the swapped face straight over glasses and
hair. There are also tests for the output shapes a face parser really emits, because
casting the wrong one to uint8 does not raise: it silently stops blending altogether.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import app.compositor as compositor_module
from app.compositor import EARS, PROTECTED_OCCLUDERS, SKIN, SemanticCompositor
from app.providers import execution_providers

SIZE = 64
CLASSES = 19
FRAME = 200
BBOX = (78, 78, 122, 122)


class FakeInput:
    def __init__(self, name="input", shape=(1, 3, SIZE, SIZE)):
        self.name, self.shape = name, shape


class FakeInferenceSession:
    """Stands in for onnxruntime so the real constructor runs without the package."""

    created: list = []

    def __init__(self, model_path, providers=None):
        self.model_path, self.providers = model_path, providers
        self.input = FakeInput()
        self.output = None
        FakeInferenceSession.created.append(self)

    def get_inputs(self):
        return [self.input]

    def run(self, _outputs, _feeds):
        return [self.output]


@pytest.fixture
def fake_ort(monkeypatch):
    module = types.ModuleType("onnxruntime")
    module.InferenceSession = FakeInferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    FakeInferenceSession.created = []
    return FakeInferenceSession


def one_hot(labels: np.ndarray) -> np.ndarray:
    """Multi-class logits, the shape a 19-class parser emits."""
    logits = np.full((1, CLASSES, SIZE, SIZE), -5.0, np.float32)
    np.put_along_axis(logits, labels[None, None], 5.0, axis=1)
    return logits


def sample_labels() -> np.ndarray:
    labels = np.zeros((SIZE, SIZE), np.uint8)
    labels[4:SIZE - 4, 4:SIZE - 4] = 1     # skin
    labels[30:33, 30:33] = 6               # glasses: protected, small enough for the close to fill
    labels[44:47, 44:47] = 0               # an ordinary hole: closing should still fill this
    return labels


def build(fake_ort, output, shape=(1, 3, SIZE, SIZE), **kwargs):
    """Construct a real SemanticCompositor against a canned parser output."""
    class Session(FakeInferenceSession):
        def __init__(self, model_path, providers=None):
            super().__init__(model_path, providers)
            self.input = FakeInput(shape=shape)
            self.output = output
    original = sys.modules["onnxruntime"].InferenceSession
    sys.modules["onnxruntime"].InferenceSession = Session
    try:
        return SemanticCompositor("face_parser.onnx", **kwargs)
    finally:
        sys.modules["onnxruntime"].InferenceSession = original


def alpha_at(alpha, row, col):
    x1, y1, _, _ = SemanticCompositor._expanded_bbox(BBOX, FRAME, FRAME)
    return float(alpha[y1 + row, x1 + col])


# ─────────────────────────── construction ───────────────────────────

def test_providers_come_from_the_registry_not_a_hardcoded_cuda_list(fake_ort):
    """A DirectML, CoreML or ROCm host must not silently parse on the CPU."""
    requested = {}
    fake_ort.created = []

    class Session(FakeInferenceSession):
        def __init__(self, model_path, providers=None):
            super().__init__(model_path, providers)
            requested["providers"] = providers

    module = sys.modules["onnxruntime"]
    original, module.InferenceSession = module.InferenceSession, Session
    try:
        SemanticCompositor("face_parser.onnx")
    finally:
        module.InferenceSession = original

    assert requested["providers"] == list(execution_providers())
    assert "CPUExecutionProvider" in requested["providers"]


def test_parser_input_size_is_read_from_the_model(fake_ort):
    assert build(fake_ort, one_hot(sample_labels())).size == SIZE
    assert build(fake_ort, one_hot(sample_labels()), shape=(1, 3, 512, 512)).size == 512


def test_a_dynamic_input_dimension_falls_back_to_512(fake_ort):
    assert build(fake_ort, one_hot(sample_labels()), shape=(1, 3, "height", "width")).size == 512


# ───────────────────── occluders are actually preserved ─────────────────────

def test_glasses_are_not_painted_over(fake_ort):
    alpha = build(fake_ort, one_hot(sample_labels())).mask(
        np.full((FRAME, FRAME, 3), 128, np.uint8), BBOX)

    skin = alpha_at(alpha, 12, 12)
    glasses = alpha_at(alpha, 31, 31)
    assert skin > 0.9
    assert glasses < skin * 0.5, f"the swapped face still covers the glasses ({glasses:.2f})"


def test_the_close_still_fills_ordinary_holes(fake_ort):
    """Excluding occluders must not disable the speckle removal it sits next to."""
    alpha = build(fake_ort, one_hot(sample_labels())).mask(
        np.full((FRAME, FRAME, 3), 128, np.uint8), BBOX)
    assert alpha_at(alpha, 45, 45) > 0.9, "a plain hole in the mask was left unfilled"


def test_the_protected_set_is_what_does_it(fake_ort, monkeypatch):
    """Proves the exclusion is the mechanism, not a coincidence of the sample layout."""
    frame = np.full((FRAME, FRAME, 3), 128, np.uint8)
    labels = one_hot(sample_labels())

    painted = build(fake_ort, labels).mask(frame, BBOX)
    monkeypatch.setattr(compositor_module, "PROTECTED_OCCLUDERS", set())
    unprotected = build(fake_ort, labels).mask(frame, BBOX)

    assert alpha_at(unprotected, 31, 31) > 0.9, "expected the old behaviour to paint over glasses"
    assert alpha_at(painted, 31, 31) < alpha_at(unprotected, 31, 31) * 0.5


def test_include_ears_toggles_ear_coverage(fake_ort):
    labels = np.zeros((SIZE, SIZE), np.uint8)
    labels[10:SIZE - 10, 10:SIZE - 10] = 1
    labels[30:33, 30:33] = sorted(EARS)[0]
    frame = np.full((FRAME, FRAME, 3), 128, np.uint8)

    with_ears = build(fake_ort, one_hot(labels), include_ears=True).mask(frame, BBOX)
    without = build(fake_ort, one_hot(labels), include_ears=False).mask(frame, BBOX)
    assert alpha_at(with_ears, 31, 31) > alpha_at(without, 31, 31)


# ──────────── parser output shapes that used to silently break ────────────

def masks(raw, classes=None, protected=None):
    return SemanticCompositor._masks(
        raw, classes or (SKIN | EARS), protected or PROTECTED_OCCLUDERS)


def test_multiclass_logits_are_argmaxed():
    blendable, protected = masks(one_hot(sample_labels()))
    assert blendable.shape == (SIZE, SIZE)
    assert blendable[12, 12] and not blendable[31, 31]
    assert protected[31, 31] and not protected[12, 12]


def test_multiclass_probabilities_are_argmaxed_too():
    logits = one_hot(sample_labels())
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    blendable, _ = masks(probabilities)
    assert blendable[12, 12] and not blendable[0, 0]


def test_single_channel_probabilities_are_thresholded_not_zeroed():
    """`astype(uint8)` turned every probability into background and the swap vanished."""
    probabilities = np.full((1, 1, SIZE, SIZE), 0.05, np.float32)
    probabilities[:, :, 10:50, 10:50] = 0.92

    blendable, protected = masks(probabilities)
    assert int(blendable.sum()) == 40 * 40, int(blendable.sum())
    assert not protected.any(), "a single-channel map carries no class information"


def test_single_channel_logits_with_negative_values_are_not_wrapped_into_class_ids():
    logits = np.full((1, 1, SIZE, SIZE), -3.2, np.float32)
    logits[:, :, 10:50, 10:50] = 2.7

    blendable, _ = masks(logits)
    assert int(blendable.sum()) == 40 * 40


def test_an_integer_label_map_is_taken_as_labels():
    labels = np.zeros((SIZE, SIZE), np.uint8)
    labels[10:50, 10:50] = 1
    labels[20:24, 20:24] = 6

    blendable, protected = masks(labels)
    assert blendable[15, 15] and protected[21, 21] and not blendable[21, 21]


def test_a_batch_dimension_of_one_is_dropped():
    blendable, _ = masks(one_hot(sample_labels())[None])
    assert blendable.shape == (SIZE, SIZE)


def test_probability_output_no_longer_discards_the_swap(fake_ort):
    """The failure this guards did not raise: blend() returned the original frame."""
    probabilities = np.full((1, 1, SIZE, SIZE), 0.02, np.float32)
    probabilities[:, :, 4:SIZE - 4, 4:SIZE - 4] = 0.95
    compositor = build(fake_ort, probabilities)

    original = np.full((FRAME, FRAME, 3), 128, np.uint8)
    swapped = original.copy()
    swapped[:] = (10, 10, 10)

    result, alpha = compositor.blend(original, swapped, BBOX)
    assert alpha.max() > 0.9, "the parser output produced no coverage at all"
    assert not np.array_equal(result, original), "the swap was silently discarded"
    x1, y1, x2, y2 = SemanticCompositor._expanded_bbox(BBOX, FRAME, FRAME)
    assert result[y1 + 20, x1 + 20].mean() < 60, "the swapped pixels did not reach the output"


def test_blend_rounds_instead_of_darkening_the_seam(fake_ort):
    compositor = build(fake_ort, one_hot(sample_labels()))
    original = np.full((FRAME, FRAME, 3), 200, np.uint8)
    swapped = np.full((FRAME, FRAME, 3), 201, np.uint8)

    result, _ = compositor.blend(original, swapped, BBOX)
    x1, y1, _, _ = SemanticCompositor._expanded_bbox(BBOX, FRAME, FRAME)
    inside = result[y1 + 12, x1 + 12].astype(int)
    assert set(inside.tolist()) <= {200, 201}
    assert inside.mean() > 200, "truncation biased the blended region downwards"
