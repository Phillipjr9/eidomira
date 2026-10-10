"""Measure how visible a face swap is, from its output alone.

"No bugs" and "realistic" are not things anyone can assert by reading code. They are
things you look at. Since the swap cannot be run without licensed weights, this measures
the artifacts that make a composite noticeable, so the claim becomes a number that can be
tracked across model changes, presets and hosts.

Three signals, all computed from an original/swapped frame pair:

* **Seam** — a discontinuity at the edge of the composited region. A hard composite shows
  a brightness or colour step along the boundary; a feathered one does not. Reported as
  the ratio of gradient energy on the boundary to gradient energy just inside it, so 1.0
  means the seam is invisible next to the texture around it.
* **Colour** — how far the swapped region's colour sits from the skin immediately around
  it. This is the other classic tell: a face pasted from a differently lit source.
* **Flicker** — frame-to-frame instability of the swapped region in excess of the real
  motion in the original frames. Shimmer is what temporal stabilisation exists to remove.

Usage::

    python -m tools.quality_report --pair original.png swapped.png
    python -m tools.quality_report --pair a1.png b1.png --pair a2.png b2.png --json

More than one pair is treated as a sequence, which is what makes flicker measurable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Pixels this far apart are treated as composited rather than compression noise.
CHANGE_THRESHOLD = 6


def changed_mask(original: np.ndarray, swapped: np.ndarray,
                 threshold: int = CHANGE_THRESHOLD) -> np.ndarray:
    """Where the two frames differ by more than `threshold` on any channel."""
    if original.shape != swapped.shape:
        raise ValueError(f"frame shapes differ: {original.shape} vs {swapped.shape}")
    delta = np.abs(original.astype(np.int16) - swapped.astype(np.int16))
    return delta.max(axis=2) > threshold


def _gradient_energy(image: np.ndarray) -> np.ndarray:
    """Per-pixel edge strength, on a luminance view so colour casts do not inflate it."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def seam_ratio(original: np.ndarray, swapped: np.ndarray,
               threshold: int = CHANGE_THRESHOLD) -> float | None:
    """Edge energy on the composited boundary relative to just inside it.

    1.0 means the boundary is no sharper than the texture it sits in. Large values mean a
    visible line; values below 1 are fine and simply indicate a soft edge.
    """
    mask = changed_mask(original, swapped, threshold)
    if not mask.any():
        return None
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(mask.astype(np.uint8), kernel) - cv2.erode(mask.astype(np.uint8), kernel)
    interior = cv2.erode(mask.astype(np.uint8), kernel)
    if not boundary.any() or not interior.any():
        return None  # the region is too small to have an inside and an edge

    energy = _gradient_energy(swapped)
    inner = float(energy[interior.astype(bool)].mean())
    outer = float(energy[boundary.astype(bool)].mean())
    if inner <= 1e-6:
        return float("inf") if outer > 1e-6 else None
    return outer / inner


def colour_shift(original: np.ndarray, swapped: np.ndarray,
                 threshold: int = CHANGE_THRESHOLD, ring: int = 6,
                 mask: np.ndarray | None = None) -> dict | None:
    """Colour distance between the composited region and the skin around it.

    The reference is a ring just outside the region rather than the whole frame, because
    what a viewer compares is the face against the jaw and neck next to it.

    `mask` pins the region being measured. Without it the region is derived from where the
    two frames differ, which makes the metric non-stationary across candidate settings: a
    stronger correction moves the region it is being measured over, so a search comparing
    two settings is comparing two different measurements. `tools/train_defaults.py` passes
    one for exactly that reason.
    """
    if mask is None:
        mask = changed_mask(original, swapped, threshold)
    if not mask.any():
        return None
    kernel = np.ones((ring, ring), np.uint8)
    surround = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
    if not surround.any():
        return None

    inside = swapped[mask].astype(np.float32).mean(axis=0)
    outside = swapped[surround].astype(np.float32).mean(axis=0)
    delta = np.abs(inside - outside)
    return {
        "region_rgb": [round(float(v), 1) for v in inside],
        "surround_rgb": [round(float(v), 1) for v in outside],
        "delta_rgb": [round(float(v), 1) for v in delta],
        "max_channel": round(float(delta.max()), 1),
        "distance": round(float(np.linalg.norm(delta)), 1),
    }


def flicker(pairs: list[tuple[np.ndarray, np.ndarray]],
            threshold: int = CHANGE_THRESHOLD) -> dict | None:
    """Frame-to-frame instability of the composited region, net of real motion.

    Real motion moves the original frames too, so it is subtracted: what remains is change
    the pipeline introduced on its own. A still subject in front of a still camera should
    score near zero no matter how much detail the model adds.
    """
    if len(pairs) < 2:
        return None

    swapped_jitter, original_jitter, region = [], [], None
    for (original_a, swapped_a), (original_b, swapped_b) in zip(pairs, pairs[1:]):
        mask = changed_mask(original_a, swapped_a, threshold)
        if not mask.any():
            continue
        if region is None:
            region = np.zeros_like(mask)
        region |= mask
        swapped_jitter.append(float(np.abs(
            swapped_b.astype(np.int16) - swapped_a.astype(np.int16))[mask].mean()))
        original_jitter.append(float(np.abs(
            original_b.astype(np.int16) - original_a.astype(np.int16))[mask].mean()))

    if not swapped_jitter:
        return None
    mean_swapped = float(np.mean(swapped_jitter))
    mean_original = float(np.mean(original_jitter))
    return {
        "frames_compared": len(swapped_jitter),
        "swapped_jitter": round(mean_swapped, 2),
        "original_jitter": round(mean_original, 2),
        "excess_flicker": round(mean_swapped - mean_original, 2),
        "reference_stddev": round(float(np.std(swapped_jitter)), 2),
    }


def assess(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """The full report for one frame, or for a sequence of them."""
    original, swapped = pairs[0]
    mask = changed_mask(original, swapped)
    report = {
        "frames": len(pairs),
        "frame_shape": list(swapped.shape),
        "composited_pixels": int(mask.sum()),
        "composited_fraction": round(float(mask.mean()), 4),
        "seam_ratio": (lambda v: None if v is None else round(v, 2))(
            seam_ratio(original, swapped)),
        "colour": colour_shift(original, swapped),
    }
    if not mask.any():
        report["verdict"] = (
            "nothing was composited — the masks are empty. A parser output format the "
            "compositor does not understand looks exactly like this."
        )
    elif len(pairs) > 1:
        report["flicker"] = flicker(pairs)
    return report


def _describe(report: dict) -> str:
    lines = [
        f"frames                 {report['frames']}",
        f"frame shape            {report['frame_shape']}",
        f"composited pixels      {report['composited_pixels']} "
        f"({report['composited_fraction'] * 100:.1f}% of the frame)",
    ]
    if report.get("verdict"):
        lines.append("")
        lines.append(f"!! {report['verdict']}")
        return "\n".join(lines)

    lines.append(f"seam ratio             {report['seam_ratio']}   (1.0 = no edge "
                 f"sharper than the texture around it)")
    colour = report.get("colour") or {}
    if colour:
        lines.append(f"colour distance        {colour['distance']}   "
                     f"(max channel {colour['max_channel']})")
        lines.append(f"  region   RGB         {colour['region_rgb']}")
        lines.append(f"  surround RGB         {colour['surround_rgb']}")
    if report.get("flicker"):
        flick = report["flicker"]
        lines.append(f"excess flicker         {flick['excess_flicker']} levels per frame "
                     f"({flick['frames_compared']} pairs)")
        lines.append(f"  swapped/original     {flick['swapped_jitter']} / "
                     f"{flick['original_jitter']} mean absolute change")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", nargs=2, action="append", metavar=("ORIGINAL", "SWAPPED"),
                        required=True, help="an original and a swapped frame; repeat for a sequence")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    pairs = []
    for original_path, swapped_path in args.pair:
        # Read first, convert second: a missing file makes imread return None, and
        # cvtColor(None) raises before any check could report which path was at fault.
        loaded = [(path, cv2.imread(str(path))) for path in (original_path, swapped_path)]
        unreadable = [str(path) for path, image in loaded if image is None]
        if unreadable:
            print(f"could not read {', '.join(unreadable)}", file=sys.stderr)
            return 2
        pairs.append(tuple(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for _, image in loaded))

    report = assess(pairs)
    print(json.dumps(report, indent=2) if args.json else _describe(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
