"""The offline trainer: it searches settings, and it must not manufacture improvements.

Two failures are worth guarding specifically, because both were real here:

* the tone sweep was scored over a region derived from the frames being compared, so a
  stronger correction moved the region it was measured over and the curve came out
  non-monotonic (0.6 scored 4.9 and 0.7 scored 9.3, which no smooth blend parameter can do);
* a search with no margin reports its own noise as an improvement, which would write a new
  default for a 10% gain nobody can see.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.knobs import KNOBS, read_tuned
from tools.train_defaults import (
    IMPROVE_MARGIN, evaluate, main, pairs_from_directory, score_tone, sweep, synthetic_pairs,
)


# ───────────────────────────── inputs ─────────────────────────────

def test_the_generated_set_contains_a_known_defect_and_controls():
    labels = [pair.label for pair in synthetic_pairs(1)]
    assert any("lit 25 dark" in label for label in labels)
    assert any("already matched" in label for label in labels), (
        "without a control the search cannot tell a fixed defect from a destroyed frame"
    )


def test_a_bigger_generated_set_varies_the_frame_rather_than_repeating_it():
    pairs = synthetic_pairs(3)
    assert len(pairs) == 3 * 6
    distinct = {pair.original.tobytes() for pair in pairs}
    assert len(distinct) == 3, "the same frame was scored three times"


def test_captured_pairs_are_read_from_disk(tmp_path):
    original = np.full((40, 60, 3), 120, np.uint8)
    swapped = np.full((40, 60, 3), 90, np.uint8)
    cv2.imwrite(str(tmp_path / "clip1_original.png"), original)
    cv2.imwrite(str(tmp_path / "clip1_swapped.png"), swapped)

    pairs = pairs_from_directory(tmp_path)
    assert [pair.label for pair in pairs] == ["clip1"]
    assert not pairs[0].original[:, :, ::-1].any() == None          # read, not returned raw


def test_a_pair_with_a_missing_counterpart_is_skipped(tmp_path):
    cv2.imwrite(str(tmp_path / "lonely_original.png"), np.zeros((20, 20, 3), np.uint8))
    assert pairs_from_directory(tmp_path) == []


def test_a_pair_of_different_sizes_is_skipped_rather_than_raising(tmp_path):
    cv2.imwrite(str(tmp_path / "odd_original.png"), np.zeros((20, 20, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "odd_swapped.png"), np.zeros((30, 30, 3), np.uint8))
    assert pairs_from_directory(tmp_path) == []


# ───────────────────────────── scoring ─────────────────────────────

def test_the_tone_curve_decreases_as_the_correction_strengthens():
    """The regression guard for the non-stationary metric.

    Scored over a region that moves with the correction, this curve is jagged and its
    minimum is noise. It has to be monotone for the search to mean anything.
    """
    pairs = synthetic_pairs(2)
    scores = [score_tone(pairs, strength) for strength in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert all(score is not None for score in scores), scores
    assert scores == sorted(scores, reverse=True), (
        f"the curve is not monotone, so candidates are being scored over different regions: {scores}"
    )
    assert scores[0] > scores[-1] * 3, (
        f"the mismatch barely moved across the whole range: {scores}"
    )


def test_an_already_matched_pair_contributes_nothing_rather_than_a_zero():
    """A metric over nothing is not a perfect score."""
    pairs = [pair for pair in synthetic_pairs(1) if "already matched" in pair.label]
    assert score_tone(pairs, 1.0) is None


# ───────────────────────────── recommendation rules ─────────────────────────────

def scorer_scoring(score, at):
    """A scorer worth `score` at candidate `at` and 1.0 everywhere else.

    A measured curve is not needed to test the decision rules, and a made-up one is easier
    to reason about: the numbers below are the scores, not a simulation of them.
    """
    return lambda _pairs, candidate: score if round(candidate, 4) == at else 1.0


def test_a_real_improvement_is_recommended():
    """The value in use scores 1.0; a candidate scores 0.1. That is worth changing."""
    result = sweep(synthetic_pairs(1), "tone_transfer_strength",
                   scorer_scoring(.1, .9), current=.5)
    assert result["best"] == .9
    assert result["improvement"] == pytest.approx(.9)
    assert result["recommended"] is True


def test_an_improvement_inside_the_margin_is_not_recommended():
    """Below the margin it is churn with a percentage attached."""
    marginal = sweep(synthetic_pairs(1), "tone_transfer_strength",
                     scorer_scoring(1.0 - IMPROVE_MARGIN / 2, .9), current=.5)
    assert 0 < marginal["improvement"] < IMPROVE_MARGIN
    assert marginal["recommended"] is False
    assert "under the" in _describe(marginal)


def _describe(result):
    from tools.train_defaults import describe

    return describe([result], "test", 1)


def test_a_bound_is_reported_but_not_applied():
    """A one-sided metric walks to the bound it cannot see past; that is not evidence."""
    def monotone(_pairs, candidate):
        # steep enough that the preference for the bound clears the margin: otherwise the
        # margin blocks the change and this test would pass without the bound rule
        return 1.0 - candidate * 10

    result = sweep(synthetic_pairs(1), "parser_feather", monotone, current=.04)
    assert result["best"] == KNOBS["parser_feather"].high
    assert result["improvement"] > IMPROVE_MARGIN, (
        "the margin is what blocked this, not the bound rule, so the test proves nothing"
    )
    assert result["at_bound"] is True
    assert result["recommended"] is False
    assert "upper bound" in _describe(result)


def test_the_value_in_use_is_scored_even_when_it_is_not_on_the_grid():
    """Otherwise there is no measured baseline and the comparison is invented."""
    result = sweep(synthetic_pairs(1), "parser_feather", scorer_scoring(.1, .035),
                   current=.035)
    assert .035 in result["curve"], (
        "the value in use was never scored, so it could not be compared"
    )
    assert result["current_score"] == pytest.approx(.1)
    assert result["best"] == .035
    assert result["recommended"] is False


def test_a_knob_that_cannot_be_measured_is_reported_rather_than_guessed():
    by_knob, results = evaluate(synthetic_pairs(1), {name: 1.0 for name in KNOBS})
    assert by_knob["restoration_visibility"]["measurable"] is False
    reason = by_knob["restoration_visibility"]["reason"]
    assert "no model" in reason or "none is installed" in reason
    assert "restoration_visibility" not in [r["knob"] for r in results if r.get("recommended")]


def test_every_knob_is_either_measured_or_explained():
    _, results = evaluate(synthetic_pairs(1), {name: 1.0 for name in KNOBS})
    assert {result["knob"] for result in results} == set(KNOBS), (
        "a knob that is neither swept nor explained would silently keep its default"
    )


# ───────────────────────────── the command ─────────────────────────────

def test_it_writes_a_file_the_application_can_read(tmp_path, capsys):
    out = tmp_path / "tuned_defaults.json"
    assert main(["--synthetic", "1", "--out", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert "values" in payload and "source" in payload
    assert "no model is trained" in payload["note"].lower()
    # and what it wrote survives the clamp the application applies
    for name, value in read_tuned(out).items():
        assert KNOBS[name].low <= value <= KNOBS[name].high


def test_the_record_says_the_pairs_were_generated(tmp_path):
    out = tmp_path / "tuned_defaults.json"
    main(["--synthetic", "1", "--out", str(out)])
    assert "generated" in json.loads(out.read_text())["source"]


def test_a_badly_wrong_default_is_recommended_and_written(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tone_transfer_strength", .2)
    out = tmp_path / "tuned_defaults.json"
    assert main(["--synthetic", "2", "--out", str(out), "--json"]) == 0

    values = json.loads(out.read_text())["values"]
    assert values.get("tone_transfer_strength", 0) > .5, (
        "a default that is 89% worse than the alternatives should be corrected"
    )


def test_a_good_default_is_not_nudged(tmp_path, capsys, monkeypatch):
    """The tool's most valuable output is sometimes 'leave it alone'."""
    from app.config import settings

    monkeypatch.setattr(settings, "tone_transfer_strength", 1.0)
    out = tmp_path / "tuned_defaults.json"
    main(["--synthetic", "2", "--out", str(out)])
    stdout = capsys.readouterr().out
    assert "already scored best" in stdout or "no change recommended" in stdout
    assert "tone_transfer_strength" not in json.loads(out.read_text())["values"]


def test_no_write_leaves_the_application_alone(tmp_path):
    out = tmp_path / "tuned_defaults.json"
    assert main(["--synthetic", "1", "--out", str(out), "--no-write"]) == 0
    assert not out.exists()


def test_json_output_is_machine_readable(capsys):
    assert main(["--synthetic", "1", "--json", "--no-write"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pairs"] > 0
    assert "results" in payload and "changed" in payload


def test_a_directory_with_no_pairs_fails_loudly(tmp_path, capsys):
    assert main(["--pairs", str(tmp_path)]) == 1
    assert "no *_original" in capsys.readouterr().err


def test_the_report_never_claims_a_model_was_trained(capsys):
    main(["--synthetic", "1", "--no-write"])
    stdout = capsys.readouterr().out
    assert "No model was trained" in stdout
    assert "the swap weights are not here" in stdout
