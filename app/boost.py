"""Recovering the detail a 128-pixel swap throws away.

`inswapper_128` works on a fixed 128x128 aligned crop. Whatever the face is in the frame —
40 pixels across or 400 — the swapped face arrives with 128x128 of detail in it, so a face
that fills a third of a 1080p frame is resampled down before it is ever pasted back. That
resampling is what "raw swaps look soft" means, and it is why every serious stack puts
something after the swapper.

Restoration is one answer (rebuild the detail with another network). This is the other, and
it needs no model at all: **sample the swap several times at sub-pixel offsets and merge the
results.**

The idea is ordinary multi-frame super-resolution. One pass of the swapper sees the source
face through a 128x128 grid. Nudge the crop by half a pixel and run it again and the second
pass sees a *different* sampling of the same face — different pixels land on the grid. Put
the shifted results back on a common grid and average them, and where a single pass had one
sample you now have four, interleaved at a spacing the single pass could not represent.

For a 2x2 grid of half-pixel offsets that doubles the effective sampling of the aligned
face: a 256x256 face from a 128x128 model. A 3x3 grid gives 384x384. Past that the model's
own output stops being a faithful sampling and the extra passes mostly average its noise
away, so the useful range is small and `SCALES` says so.

Two honest limits, both of which matter:

* What is proven here is the **mechanism**: phased sampling, grid arithmetic, the inverse
  warp, and that merging phase-shifted samples genuinely reconstructs detail a single pass
  loses. `tests/test_boost.py` measures it against a resolution-limited stand-in with a
  known ground truth.
* What is **not** proven is how much a real generative swapper gains, because no swap model
  has ever run on this machine. A network's output is not a plain resampling of its input,
  so the gain is expected to be smaller than the ideal case. Treat the boost as "more detail
  than the raw 128 output, at N times the swap cost" until `tools/quality_report.py` can be
  pointed at a licensed model and a real pair.
"""
from __future__ import annotations

import cv2
import numpy as np

#: The aligned crop size `inswapper_128` is fixed to. The whole module is built around it.
ALIGN = 128

#: Multipliers worth offering. 2x and 3x are the range where phased sampling still buys
#: detail; 4x costs sixteen passes for a face the model cannot resolve that finely.
SCALES = (1, 2, 3)


def phases(scale: int) -> list[tuple[int, int]]:
    """The sub-pixel offsets for a `scale`x boost, in canvas pixels.

    A 2x boost is a 2x2 grid of half-pixel offsets, which after the 2x upscale onto the
    canvas land on integer canvas pixels: 0 and 1. That is the whole trick — the offsets
    the swapper cannot express become exact integers once the canvas is twice as wide.
    """
    scale = int(scale)
    if scale < 1:
        raise ValueError(f"scale must be at least 1, got {scale}")
    return [(i, j) for i in range(scale) for j in range(scale)]


def crop_transform(base: np.ndarray, scale: int) -> np.ndarray:
    """The source-to-crop transform, expressed for the `scale`x canvas.

    The canvas is the aligned crop at `scale` times the size, so both the linear part and
    the translation scale by the same factor. Transforms here are cv2's 2x3 affines
    throughout: three-by-three matrices with a fixed last row are a needless way to be
    wrong, and cv2 rejects them anyway.
    """
    scale = int(scale)
    return np.asarray(base, np.float64) * scale


def shifted_transform(base: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """`base`, adjusted so the cropped content moves by (dx, dy) crop pixels.

    `cv2.warpAffine(src, M, (128, 128))` samples `src` at `M⁻¹` of each output pixel, so
    moving the destination by -d moves the content by +d.
    """
    moved = np.array(base, np.float64, copy=True)
    if moved.shape != (2, 3):
        raise ValueError(f"expected a 2x3 affine, got {moved.shape}")
    moved[0, 2] -= dx
    moved[1, 2] -= dy
    return moved


def merge(results: list[tuple[tuple[int, int], np.ndarray]], scale: int) -> np.ndarray:
    """Interleave phase-shifted swap results onto one `scale`x canvas.

    Not an average. `scale`² passes of `ALIGN`² carry exactly `(ALIGN * scale)`² samples —
    four 128 squares are one 256 square, sample for sample — so every canvas pixel has
    exactly one pass that measured it and nothing is being estimated. Averaging instead
    means upscaling each pass and blending the interpolated results, which smooths away the
    very high frequencies the extra passes were run to capture; measured against a known
    ground truth it scored *worse* than a single pass, because the interpolation cost more
    detail than the phases added.

    A phase is (horizontal, vertical) — the order `warp_crop` takes — and rows are indexed
    first, so the vertical phase picks the row stride and the horizontal one the column
    stride. Mixing those two up costs about 4 dB and looks like a sharpening filter, because
    the passes land one pixel out and interleave along the wrong axis.
    """
    scale = int(scale)
    if scale < 2:
        raise ValueError("merge is for scale >= 2; a single pass needs nothing merged")
    if len(results) != scale * scale:
        raise ValueError(f"a {scale}x boost needs {scale * scale} passes, got {len(results)}")
    size = ALIGN * scale
    canvas = np.zeros((size, size, 3), np.float32)
    seen = np.zeros((size, size, 1), bool)
    for (x_phase, y_phase), result in results:
        if result.shape[0] != ALIGN or result.shape[1] != ALIGN:
            raise ValueError(f"a swap pass must return {ALIGN}x{ALIGN}, got {result.shape}")
        if not (0 <= x_phase < scale and 0 <= y_phase < scale):
            raise ValueError(f"phase ({x_phase}, {y_phase}) is outside a {scale}x grid")
        if seen[y_phase::scale, x_phase::scale].any():
            raise ValueError(f"two passes claim phase ({x_phase}, {y_phase})")
        canvas[y_phase::scale, x_phase::scale] = result
        seen[y_phase::scale, x_phase::scale] = True
    # No coverage check is needed, and an unreachable one would be worse than none: with
    # scale squared distinct in-range phases, every residue class is used exactly once, so
    # the count, range and duplicate guards above are what makes the canvas whole.
    return np.rint(canvas).clip(0, 255).astype(np.uint8)


def boosted_face(warp_crop, swap_once, scale: int = 2) -> np.ndarray:
    """The swapped face as a `scale`*128 canvas, from `scale`² passes of the swapper.

    Both callables belong to the caller so this stays pure geometry:

    * `warp_crop(dx, dy)` returns the `ALIGN` crop of the source with its content moved by
      (dx, dy) crop pixels — the engine's alignment code, half a pixel over.
    * `swap_once(crop)` returns the model's `ALIGN` output for one crop.

    `scale=1` is the unboosted path: one pass at zero offset, no merge.
    """
    scale = int(scale)
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale}")
    if scale == 1:
        return np.asarray(swap_once(warp_crop(0.0, 0.0)), np.uint8)

    results = []
    for x_phase, y_phase in phases(scale):
        # The offsets are in *crop* pixels, and a crop pixel is 1/scale of a canvas pixel,
        # so a phase of one canvas pixel is 1/scale of a crop pixel.
        crop = warp_crop(x_phase / scale, y_phase / scale)
        results.append(((x_phase, y_phase), np.asarray(swap_once(crop), np.uint8)))
    return merge(results, scale)


def paste_back(frame_bgr: np.ndarray, face_bgr: np.ndarray, base: np.ndarray,
               scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp a boosted face into the frame. Returns (frame, alpha) in BGR/aligned space.

    The inverse of the transform that produced the canvas, with an alpha that is one where
    the canvas really landed. The swapper's own `paste_back` cannot be used here: it knows
    about the 128 crop, not the canvas, and it pastes with a hard mask — this keeps the
    output in the pipeline's own compositing path so the occluder-preserving mask and the
    tone transfer still apply to a boosted face.
    """
    height, width = frame_bgr.shape[:2]
    transform = crop_transform(base, scale)
    inverse = cv2.invertAffineTransform(transform)
    # Constant, not replicate: this layer is only valid where `alpha` is one, and an edge
    # smeared outwards by the border mode would put face pixels where the mask says there
    # are none, which is the one kind of mistake the compositing path exists to avoid.
    warped = cv2.warpAffine(face_bgr, inverse, (width, height), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones(frame_bgr.shape[:2], np.float32)
    alpha = cv2.warpAffine(ones, inverse, (width, height), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, alpha
