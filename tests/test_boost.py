"""Proof that phased sampling recovers detail a single 128-pixel pass cannot hold.

The stand-in for the swapper is a process that can only ever see the face through a 128x128
grid, and returns what it saw unchanged. That isolates the one property this module exists
to exploit: a single pass has 128 samples across the face, and four passes at half-pixel
offsets have 256 — measured, interleaved, not estimated.

What is proven here is the mechanism. What it buys against a real generative swapper is not
proven, because no swap model has ever run on this machine: a network's output is not a
plain resampling of its input. `tools/quality_report.py` on a licensed model is the test
that settles that, and until it has run, the honest claim is "more detail than the raw 128
output, for `scale`² times the swap cost".
"""
import cv2
import numpy as np
import pytest

from app.boost import (ALIGN, SCALES, boosted_face, crop_transform, merge, paste_back,
                       phases, shifted_transform)


# ───────────────────────────────── the stand-in ─────────────────────────────────

def detailed_source(size: int = 512) -> np.ndarray:
    """A frame whose face region holds detail finer than the 128 grid can carry.

    Three patterns, chosen so a wrong answer is obvious rather than merely slightly wrong:
    a checkerboard exactly at the 128 grid's Nyquist limit, plus strong vertical and
    horizontal lines at different periods. A transposed or misaligned merge cannot
    reconstruct it by accident.
    """
    image = np.full((size, size, 3), 90, np.uint8)
    face = np.full((256, 256, 3), 170, np.uint8)
    checkerboard = np.indices((256, 256)).sum(0) % 2
    face[checkerboard == 1] = np.array([120, 150, 190], np.uint8)
    face[::7, :] = np.array([40, 60, 80], np.uint8)
    face[:, ::11] = np.array([200, 190, 60], np.uint8)
    image[128:384, 128:384] = face
    return image


X0 = Y0 = 128
#: The face is 256 px across in the frame and the crop is 128, so the crop is at half
#: scale — exactly the situation the real swapper is in when the face fills much of a frame.
SCALE_HALF = np.array([[.5, 0, -.5 * X0], [0, .5, -.5 * Y0]], np.float64)


def resolution_limited_swapper(source: np.ndarray):
    """Returns (warp_crop, swap_once) for a process limited to a 128-pixel grid."""

    def warp_crop(dx, dy):
        return cv2.warpAffine(
            source, shifted_transform(SCALE_HALF, dx, dy), (ALIGN, ALIGN),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

    return warp_crop, lambda crop: crop


def psnr(a, b) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 10 * np.log10(255.0 ** 2 / max(1e-9, np.mean((a - b) ** 2)))


def sharpness(image) -> float:
    grey = cv2.cvtColor(np.ascontiguousarray(image.astype(np.uint8)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


# ─────────────────────────── the claim, measured ───────────────────────────

def test_the_boost_reconstructs_the_aligned_crop_at_scale_times_size():
    """The mechanism, as an identity rather than a similarity.

    `scale`² passes of a 128-pixel process, interleaved, are *exactly* the aligned crop
    rendered at `scale * 128` — bit for bit, at every scale. Not "sharper", not "close":
    equal. That is what makes this a resolution recovery rather than a sharpening filter.

    It is also the test that caught the interleave indexing rows with the horizontal
    phase: those all-exact samples still scored 15 dB against the truth, and this
    comparison failed by exactly the transpose.
    """
    source = detailed_source()
    warp_crop, swap_once = resolution_limited_swapper(source)

    for scale in (2, 3):
        size = ALIGN * scale
        expected = cv2.warpAffine(
            source, crop_transform(SCALE_HALF, scale), (size, size),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        boosted = boosted_face(warp_crop, swap_once, scale)
        assert boosted.shape == (size, size, 3)
        assert np.array_equal(boosted, expected), (
            f"{scale}x: mean error {np.abs(boosted.astype(int) - expected.astype(int)).mean():.2f}"
        )


def test_the_boost_beats_one_pass_on_the_detail_a_pass_cannot_hold():
    """Why the module exists, measured against a single pass on the same stand-in."""
    source = detailed_source()
    truth = source[Y0:Y0 + 256, X0:X0 + 256]
    warp_crop, swap_once = resolution_limited_swapper(source)

    single = swap_once(warp_crop(0, 0))
    upscaled = cv2.resize(single, (256, 256), interpolation=cv2.INTER_CUBIC)
    boosted = boosted_face(warp_crop, swap_once, 2)

    assert psnr(upscaled, truth) < 20, "the stand-in should really be losing the detail"
    assert psnr(boosted, truth) > 60, "the boost should really be recovering it"
    assert sharpness(boosted) > 5 * sharpness(upscaled)


def test_one_pass_is_unchanged_by_being_boosted():
    """scale=1 must be the old path exactly, not an approximation of it."""
    source = detailed_source()
    warp_crop, swap_once = resolution_limited_swapper(source)

    unboosted = boosted_face(warp_crop, swap_once, 1)

    assert np.array_equal(unboosted, swap_once(warp_crop(0, 0)))
    assert unboosted.shape == (ALIGN, ALIGN, 3)


def test_a_three_by_three_grid_is_nine_passes_over_a_larger_canvas():
    """3x is a bigger canvas, not a sharper one, and saying so keeps the setting honest.

    With the face at half scale in the frame, a 2x canvas is already 1:1 with the source,
    so 3x is sampling between the source's own pixels and gains nothing. It is worth the
    extra passes when the face is *smaller* than the crop, where finer phases do carry new
    information — the caller decides, from the face size, and this only reports the shape.
    """
    source = detailed_source()
    warp_crop, swap_once = resolution_limited_swapper(source)

    two = boosted_face(warp_crop, swap_once, 2)
    three = boosted_face(warp_crop, swap_once, 3)

    assert two.shape == (256, 256, 3)
    assert three.shape == (384, 384, 3)
    # Both are exact renderings of the same crop, so comparing their sharpness would only
    # compare their sizes. What must hold is that neither is an interpolation of the other.
    assert not np.array_equal(cv2.resize(two, (384, 384), interpolation=cv2.INTER_CUBIC), three)


def test_the_interleave_neither_averages_nor_amplifies_a_passs_noise():
    """Stated as a test so it stays true, and so nobody expects the boost to clean up.

    An interleave gives every output pixel exactly one pass's sample, so it cannot average
    a pass's own noise away: the output noise measures the same as the input noise, to
    within rounding. That is the reason the gain on a real generative swapper will be
    smaller than the exact case above, and the reason restoration still has a job.
    """
    source = detailed_source()
    warp_crop, _ = resolution_limited_swapper(source)
    rng = np.random.default_rng(11)
    noise = rng.integers(-12, 13, 400_000)

    def noisy_swapper(crop):
        return np.clip(crop.astype(np.int16) + rng.integers(-12, 13, crop.shape),
                       0, 255).astype(np.uint8)

    boosted = boosted_face(warp_crop, noisy_swapper, 2)

    residual = boosted.astype(np.float64) - source[Y0:Y0 + 256, X0:X0 + 256]
    assert abs(residual.std() - noise.std()) < 1.0, (
        f"output noise {residual.std():.2f} against input noise {noise.std():.2f}: an "
        "interleave should pass a pass's noise through unchanged"
    )


# ─────────────────────────────── the guards ───────────────────────────────

def test_a_phase_grid_covers_every_square():
    assert phases(1) == [(0, 0)]
    assert phases(2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(phases(3)) == 9
    assert len(set(phases(3))) == 9


@pytest.mark.parametrize("scale", [0, -1])
def test_a_scale_below_one_is_refused(scale):
    with pytest.raises(ValueError, match="at least 1"):
        phases(scale)


@pytest.mark.parametrize("scale", [4, 5, 8, 0])
def test_a_scale_the_model_cannot_resolve_is_refused(scale):
    """Beyond 3x the model's output stops being a sampling of the face, and the extra passes
    cost scale² swaps. Refusing is better than a setting that is quietly useless."""
    assert scale not in SCALES
    with pytest.raises(ValueError, match="one of"):
        boosted_face(lambda dx, dy: np.zeros((ALIGN, ALIGN, 3), np.uint8),
                     lambda crop: crop, scale)


def test_too_few_passes_for_the_scale_is_refused():
    """Four 128-squares are one 256-square. Three of them are not, and the missing quarter
    is a hole in the face rather than something to fill in with a guess."""
    square = np.zeros((ALIGN, ALIGN, 3), np.uint8)
    with pytest.raises(ValueError, match="needs 4 passes, got 2"):
        merge([((0, 0), square), ((1, 1), square)], 2)
    with pytest.raises(ValueError, match="needs 9 passes, got 8"):
        merge([((i, j), square) for i in range(3) for j in range(3)][:8], 3)


def test_two_passes_cannot_claim_the_same_phase():
    square = np.zeros((ALIGN, ALIGN, 3), np.uint8)
    results = [((0, 0), square)] * 4
    with pytest.raises(ValueError, match="two passes claim phase"):
        merge(results, 2)


def test_a_phase_outside_the_grid_is_refused():
    square = np.zeros((ALIGN, ALIGN, 3), np.uint8)
    with pytest.raises(ValueError, match="outside a 2x grid"):
        merge([((0, 0), square), ((0, 1), square), ((1, 0), square), ((2, 2), square)], 2)


def test_a_pass_that_returns_the_wrong_size_is_refused():
    """InsightFace returns 128 for inswapper_128. A 256 return would mean the model is not
    the one this module was built for, and interleaving it would scramble the face."""
    good = np.zeros((ALIGN, ALIGN, 3), np.uint8)
    with pytest.raises(ValueError, match="must return 128x128"):
        merge([((0, 0), good), ((0, 1), good), ((1, 0), good),
               ((1, 1), np.zeros((256, 256, 3), np.uint8))], 2)


def test_merge_refuses_a_single_pass():
    with pytest.raises(ValueError, match="scale >= 2"):
        merge([((0, 0), np.zeros((ALIGN, ALIGN, 3), np.uint8))], 1)


# ─────────────────────────────── the geometry ───────────────────────────────

def test_a_shifted_crop_moves_the_content_the_way_the_merge_assumes():
    """The one assumption merge and boosted_face share, asserted on its own so a change to
    the sign here fails next to the code that depends on it rather than three tests away."""
    source = detailed_source()
    warp_crop, _ = resolution_limited_swapper(source)
    truth = source[Y0:Y0 + 256, X0:X0 + 256]

    # A phase of half a crop pixel is a whole canvas pixel at 2x, and a horizontal phase
    # moves *columns*. Indexing rows here is the transpose that cost 4 dB in merge.
    for canvas_offset in (0, 1):
        crop = warp_crop(canvas_offset / 2, 0)
        assert np.array_equal(crop, truth[::2, canvas_offset::2]), (
            f"phase {canvas_offset} did not sample canvas columns {canvas_offset}::2"
        )


def test_the_canvas_transform_is_the_crop_transform_at_scale():
    assert np.allclose(crop_transform(SCALE_HALF, 1), SCALE_HALF)
    assert np.allclose(crop_transform(SCALE_HALF, 2), SCALE_HALF * 2)


def test_a_three_by_three_matrix_is_refused_rather_than_silently_wrong():
    three_by_three = np.eye(3)
    with pytest.raises(ValueError, match="2x3 affine"):
        shifted_transform(three_by_three, .5, .5)


def test_pasting_a_boosted_face_back_lands_where_the_transform_says():
    """The inverse warp, checked against a marker rather than by eye."""
    frame = np.full((512, 512, 3), 90, np.uint8)
    face = np.zeros((256, 256, 3), np.uint8)
    face[64:192, 64:192] = np.array([10, 200, 30], np.uint8)

    warped, alpha = paste_back(frame, face, SCALE_HALF, 2)

    assert warped.shape == frame.shape
    # The canvas is 1:1 with the source region, so canvas pixel (c, c) is frame
    # (Y0 + c, X0 + c) exactly.
    assert np.array_equal(warped[Y0 + 100, X0 + 100], face[100, 100])
    assert alpha[Y0 + 100, X0 + 100] == 1
    # Outside the canvas the layer is zero and the mask says so, so the caller compositing
    # with `alpha` cannot paint face pixels there by accident.
    assert alpha[10, 10] == 0
    assert np.array_equal(warped[10, 10], np.zeros(3, np.uint8))

def test_pasting_a_boosted_face_back_lands_where_the_transform_says():
    """The inverse warp, checked against a marker rather than by eye."""
    frame = np.full((512, 512, 3), 90, np.uint8)
    face = np.zeros((256, 256, 3), np.uint8)
    face[64:192, 64:192] = np.array([10, 200, 30], np.uint8)

    warped, alpha = paste_back(frame, face, SCALE_HALF, 2)

    assert warped.shape == frame.shape
    # The canvas is 1:1 with the source region, so canvas pixel (c, c) is frame
    # (Y0 + c, X0 + c) exactly.
    assert np.array_equal(warped[Y0 + 100, X0 + 100], face[100, 100])
    assert alpha[Y0 + 100, X0 + 100] == 1
    # Outside the canvas the layer is zero and the mask says so, so a caller compositing
    # with `alpha` cannot paint face pixels there by accident.
    assert alpha[10, 10] == 0
    assert np.array_equal(warped[10, 10], np.zeros(3, np.uint8))


def test_the_pasted_layer_matches_the_alignment_it_was_given():
    """A frame-sized read of the same transform: the face lands on the source's coordinates,
    so a face swapped at half scale comes back at half scale and not offset."""
    frame = np.full((512, 512, 3), 90, np.uint8)
    face = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (256, 1, 3))

    warped, alpha = paste_back(frame, face, SCALE_HALF, 2)

    valid = alpha > .5
    assert valid.any()
    rows, cols = np.nonzero(valid)
    assert rows.min() >= Y0 - 1 and cols.min() >= X0 - 1
    # Column c of the canvas comes back at frame column X0 + c, one to one.
    assert np.array_equal(warped[Y0 + 40, X0:X0 + 256, 0], face[40, :, 0])
