"""The quality metrics themselves, against cases whose answers are known in advance.

A metric that has never been checked against a case it must score badly is just a number
generator. Each test here builds a frame pair whose defect is known by construction —
a hard seam, a colour cast, added jitter, real motion — and checks the metric reacts.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from tools.quality_report import (
    assess,
    changed_mask,
    colour_shift,
    flicker,
    main,
    seam_ratio,
)

H, W = 200, 260
SKIN = (slice(50, 150), slice(80, 180))
#: A uniform 25-level offset per channel. Euclidean distance per channel is 25, so the
#: total is 25*sqrt(3) = 43.3 — a value the colour metric should reproduce.
COLOUR_OFFSET = 25


def _texture(step, value, seed_shift=0):
    texture = np.zeros((H, W, 3), np.int16)
    texture[::3, ::3] = value
    texture[1::5, 2::5] = -value // 2
    return texture


def scene():
    """A deterministic, textured frame. No per-frame randomness: jitter must be ours."""
    return np.clip(168 + _texture(3, 12), 0, 255).astype(np.uint8)


def uniform_offset(offset: int):
    """The scene shifted by one uniform offset: colour moves, texture does not.

    Isolating the two matters, because texture difference is not a colour error and a
    metric that conflates them cannot tell a badly lit face from a detailed one.
    """
    return np.clip(scene().astype(np.int16) + offset, 0, 255).astype(np.uint8)


def texture_only():
    """The scene blurred: the same average colour, none of its texture."""
    return cv2.GaussianBlur(scene(), (9, 9), 0)


def composite(alpha_width: int, offset: int = 0, face_frame: np.ndarray | None = None):
    """Put a face into the scene, with a hard edge or a feathered one.

    `alpha_width` of 0 is a hard cut; 21 is the sort of feather the compositor produces.
    """
    original = scene()
    incoming = uniform_offset(offset) if face_frame is None else face_frame
    alpha = np.zeros((H, W), np.float32)
    alpha[SKIN] = 1.0
    if alpha_width:
        alpha = cv2.GaussianBlur(alpha, (alpha_width | 1, alpha_width | 1), 0)
    a = alpha[..., None]
    blended = original.astype(np.float32) * (1 - a) + incoming.astype(np.float32) * a
    return np.rint(np.clip(blended, 0, 255)).astype(np.uint8)


#: The face sits this far from the scene colour, so a sequence always has a face to
#: measure. Without it, jitter=0 would composite nothing and there would be no region.
FACE_BASE = 20


def sequence(jitter: int, motion: int, frames: int = 6):
    """Frame pairs where the face travels with the scene, so motion is not a defect."""
    pairs = []
    for index in range(frames):
        shift = index * motion
        original = np.roll(scene(), shift, axis=1)
        offset = FACE_BASE + (jitter if index % 2 else -jitter)
        frame = composite(21, offset=offset)
        pairs.append((original, np.roll(frame, shift, axis=1)))
    return pairs


# ───────────────────────────── composited region ─────────────────────────────

def test_an_untouched_frame_reports_nothing_composited():
    report = assess([(scene(), scene())])
    assert report["composited_pixels"] == 0
    assert "nothing was composited" in report["verdict"]
    assert report["seam_ratio"] is None


def test_the_composited_region_is_found():
    mask = changed_mask(scene(), composite(0, offset=8))
    assert mask.any()
    # the face box is 100x100 of a 200x260 frame
    assert 0.15 < mask.mean() < 0.25


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError):
        changed_mask(scene(), scene()[:100])


def test_a_sub_threshold_difference_counts_as_untouched():
    almost = scene().copy()
    almost[SKIN] = np.clip(almost[SKIN].astype(int) + 3, 0, 255).astype(np.uint8)
    assert not changed_mask(scene(), almost).any()


# ───────────────────────────────── the seam ─────────────────────────────────

def test_a_hard_seam_scores_far_worse_than_a_feathered_one():
    hard = seam_ratio(scene(), composite(0, offset=-COLOUR_OFFSET))
    soft = seam_ratio(scene(), composite(21, offset=-COLOUR_OFFSET))
    assert hard > 2.0, f"a hard edge should stand out against the texture, got {hard}"
    assert soft < hard / 2, f"feathering should soften the boundary: {soft} vs {hard}"


def test_a_well_composited_face_scores_well():
    """A feathered, close-in-colour face: no boundary and nothing to flag."""
    good = composite(21, offset=8)
    assert changed_mask(scene(), good).any(), "the face should still be composited"
    assert seam_ratio(scene(), good) < 1.2
    assert colour_shift(scene(), good)["distance"] < 6


def test_a_texture_difference_is_not_reported_as_a_colour_error():
    """Blurring preserves the average colour, so only texture changed."""
    recoloured = composite(21, face_frame=texture_only())
    assert changed_mask(scene(), recoloured).any()
    assert colour_shift(scene(), recoloured)["distance"] < 3


def test_an_untouched_frame_reports_no_seam_at_all():
    assert seam_ratio(scene(), scene()) is None


# ───────────────────────────────── colour ─────────────────────────────────

def test_colour_shift_reproduces_a_known_offset():
    """Hard-cut the face in, so the region's average really is offset by COLOUR_OFFSET.

    With a feather the mask includes the partly blended boundary, which pulls the average
    towards the surroundings and is a property of the composite, not of the metric.
    """
    shift = colour_shift(scene(), composite(0, offset=-COLOUR_OFFSET))
    assert shift["max_channel"] == pytest.approx(COLOUR_OFFSET, abs=3)
    assert shift["distance"] == pytest.approx(COLOUR_OFFSET * np.sqrt(3), abs=6)


def test_a_colour_matched_face_reads_as_matched():
    shift = colour_shift(scene(), composite(21, face_frame=texture_only()))
    assert shift["distance"] < 3, shift


def test_the_comparison_is_against_the_surroundings_not_the_whole_frame():
    """A global reference would hide a face that is wrong only next to its own jaw."""
    shift = colour_shift(scene(), composite(21, offset=-COLOUR_OFFSET))
    assert shift["region_rgb"] != shift["surround_rgb"]


# ──────────────────────────────── flicker ────────────────────────────────

def test_a_perfectly_still_sequence_has_no_excess_flicker():
    assert flicker(sequence(0, 0))["excess_flicker"] == 0.0


def test_added_frame_to_frame_jitter_is_reported():
    """Alternating +/-8 must show as roughly 16 levels of change in the face region.

    Not exactly 16: the mask includes the feathered boundary, where the blend is only
    partial, so the average over the region is pulled slightly below the peak.
    """
    result = flicker(sequence(8, 0))
    assert 12 < result["excess_flicker"] < 17, result
    assert result["original_jitter"] == 0.0, "the scenes themselves are still"


def test_real_motion_is_cancelled_out():
    """Motion moves the originals too, so it is not the pipeline's fault."""
    still, moving = flicker(sequence(0, 0)), flicker(sequence(0, 3))
    assert abs(moving["excess_flicker"]) < 3, moving
    assert moving["swapped_jitter"] > still["swapped_jitter"]


def test_jitter_is_still_visible_on_top_of_motion():
    assert flicker(sequence(8, 3))["excess_flicker"] > 12


def test_a_single_pair_has_no_flicker():
    assert flicker([(scene(), composite(21))]) is None
    assert "flicker" not in assess([(scene(), composite(21))])


# ────────────────────────────────── report ──────────────────────────────────

def test_the_report_covers_the_frame_and_the_sequence():
    report = assess(sequence(8, 0))
    assert report["frames"] == 6
    assert report["frame_shape"] == [H, W, 3]
    assert report["seam_ratio"] is not None
    assert report["colour"] is not None
    assert report["flicker"]["excess_flicker"] > 10


def test_command_line_writes_a_report(tmp_path, capsys):
    original_path, swapped_path = tmp_path / "original.png", tmp_path / "swapped.png"
    cv2.imwrite(str(original_path), cv2.cvtColor(scene(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(swapped_path), cv2.cvtColor(composite(0, -COLOUR_OFFSET), cv2.COLOR_RGB2BGR))

    assert main(["--pair", str(original_path), str(swapped_path)]) == 0
    text = capsys.readouterr().out
    assert "seam ratio" in text and "colour distance" in text


def test_command_line_emits_json(tmp_path, capsys):
    original_path, swapped_path = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(original_path), cv2.cvtColor(scene(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(swapped_path), cv2.cvtColor(composite(0, offset=8), cv2.COLOR_RGB2BGR))

    assert main(["--pair", str(original_path), str(swapped_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["composited_pixels"] > 0


def test_command_line_reports_unreadable_files(tmp_path, capsys):
    assert main(["--pair", str(tmp_path / "missing.png"), str(tmp_path / "also.png")]) == 2
    assert "could not read" in capsys.readouterr().err
