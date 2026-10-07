"""The trainer: detection, bounded tuning, verification, and the report.

Thresholds are calibrated here rather than assumed, because they are only meaningful at
`MEASURE_WIDTH`. If the measurement size changes, `test_the_thresholds_still_separate_healthy_from_broken`
is what fails, instead of the trainer quietly mislabelling every frame.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from app.knobs import KNOBS, FIXED, Knob, apply_to_settings, read_tuned
from app.trainer import (
    COLOUR_LIMIT, MEASURE_WIDTH, NO_CHANGE_LIMIT, SEAM_LIMIT, Trainer,
)
from test_quality_report import composite, scene, texture_only

# ─────────────────────────────── measurement ───────────────────────────────

def measured(frame, trainer=None):
    """Run one pair through the trainer's own measurement."""
    return (trainer or Trainer(sample_every=1)).measure(scene(), frame)


def test_the_thresholds_still_separate_healthy_from_broken():
    """The calibration, at the width the trainer actually measures at.

    Without this, a change to MEASURE_WIDTH or to the metrics would leave the thresholds
    separating nothing, and every frame would be either always fine or always broken.
    """
    healthy = measured(texture_only())
    mild = measured(composite(21, offset=6))
    mismatch = measured(composite(21, offset=-25))
    hard = measured(composite(0, offset=-25))

    # healthy pairs are below both limits...
    assert healthy["colour"] < COLOUR_LIMIT
    assert mild["seam"] is None or mild["seam"] < SEAM_LIMIT
    # ...and a real lighting mismatch is above the colour limit
    assert mismatch["colour"] > COLOUR_LIMIT * 1.5, mismatch
    # ...and a hard cut is above the seam limit
    assert hard["seam"] > SEAM_LIMIT * 1.5, hard
    # a feathered composite of the same face is not
    assert mismatch["seam"] is None or mismatch["seam"] < SEAM_LIMIT


def test_measurement_is_downscaled_and_still_reports_a_known_defect():
    big = np.repeat(np.repeat(scene(), 4, axis=0), 4, axis=1)
    small_original = big
    small_swapped = np.repeat(np.repeat(composite(21, offset=-25), 4, axis=0), 4, axis=1)
    metrics = Trainer(sample_every=1).measure(small_original, small_swapped)
    assert metrics["colour"] > COLOUR_LIMIT


def test_measurement_survives_a_bigger_frame_than_the_measure_width():
    """The size used for measurement is fixed, so cost does not scale with camera size."""
    trainer = Trainer(sample_every=1, measure_width=MEASURE_WIDTH)
    for scale in (1, 4, 8):
        frame = np.repeat(np.repeat(scene(), scale, axis=0), scale, axis=1)
        assert trainer.measure(frame, frame) is not None


def test_measurement_runs_on_a_small_frame_whatever_the_camera_sends(monkeypatch):
    """The invariant behind the cost, asserted as a shape rather than a stopwatch.

    A full-resolution sample measured 64 ms at 960x738 on this hardware — more than the
    whole frame budget — so `measure` downscales first. An earlier version of this test
    asserted a millisecond figure and failed at 17 ms on a loaded machine while saying
    nothing about whether the downscale was still there. This says exactly that.
    """
    import tools.quality_report as quality_report

    measured: list[tuple[int, int]] = []
    real_seam_ratio = quality_report.seam_ratio

    def watching_seam_ratio(original, *args, **kwargs):
        measured.append(original.shape[:2])
        return real_seam_ratio(original, *args, **kwargs)

    monkeypatch.setattr(quality_report, "seam_ratio", watching_seam_ratio)

    trainer = Trainer(sample_every=1)
    for scale in (1, 4, 8):                      # 260x200 up to 2080x1600
        frame = np.repeat(np.repeat(scene(), scale, axis=0), scale, axis=1)
        trainer.measure(frame, frame)

    assert len(measured) == 3
    assert all(width <= MEASURE_WIDTH for _, width in measured), (
        f"the metrics were handed {measured}, so their cost scales with the camera"
    )


def test_measurement_refuses_a_pair_it_cannot_compare():
    trainer = Trainer(sample_every=1)
    assert trainer.measure(scene(), scene()[:100]) is None


def test_an_identical_pair_reports_no_change():
    trainer = Trainer(sample_every=1)
    metrics = trainer.measure(scene(), scene())
    assert metrics["mean_abs"] < NO_CHANGE_LIMIT


# ─────────────────────────────── detection ───────────────────────────────

def observe(trainer, frame=30, **metrics):
    trainer.observe(frame, metrics={"seam": None, "colour": None, "mean_abs": 20.0, **metrics})


def test_a_colour_mismatch_is_recorded_with_its_measurement():
    trainer = Trainer(sample_every=1)
    observe(trainer, colour=31.9)
    assert trainer.count("colour_mismatch") == 1
    mistake = trainer.mistakes[0]
    assert mistake.severity == "warning"
    assert "another room" in mistake.summary
    assert mistake.evidence["colour"] == 31.9


def test_a_visible_edge_is_recorded_with_its_measurement():
    trainer = Trainer(sample_every=1)
    observe(trainer, colour=4.0, seam=7.8)
    assert trainer.count("visible_edge") == 1
    assert trainer.mistakes[0].evidence["seam"] == 7.8


def test_a_frame_that_claimed_a_face_but_did_not_change_is_a_defect():
    """The silent no-op signature, which is how the compositor bug hid for so long."""
    trainer = Trainer(sample_every=1)
    observe(trainer, colour=None, seam=None, mean_abs=0.0)
    assert trainer.count("no_change") == 1
    assert "did nothing" in trainer.mistakes[0].summary


def test_repeats_are_counted_without_flooding_the_ledger():
    trainer = Trainer(sample_every=1)
    for _ in range(9):
        observe(trainer, colour=31.9)
    assert trainer.count("colour_mismatch") == 9
    assert len(trainer.mistakes) == 1, "the ledger should hold one entry per kind"


def test_a_healthy_frame_records_nothing():
    trainer = Trainer(sample_every=1)
    observe(trainer, colour=4.0, seam=1.2)
    assert trainer.counts == {}


def test_verification_frames_are_not_judged_on_quality():
    """During the liveness check the processor deliberately passes the frame through;
    calling that a silent no-op would report a defect every session."""
    trainer = Trainer(sample_every=1)
    trainer.observe(12, metrics={"colour": None, "seam": None, "mean_abs": 0.0},
                    steady_state=False)
    assert trainer.count("no_change") == 0


def test_no_face_is_not_a_defect():
    trainer = Trainer(sample_every=1)
    trainer.observe(12, metrics={"colour": None, "seam": None, "mean_abs": 0.0},
                    face_found=False)
    assert trainer.counts == {}


def test_latency_is_a_defect_only_after_a_run_of_frames():
    trainer = Trainer(sample_every=1)
    for frame in range(1, 4):
        trainer.observe(frame, frame_ms=90.0, target_ms=45.0)
    assert trainer.count("over_budget") == 0, "one slow frame is jitter, not a defect"
    trainer.observe(4, frame_ms=90.0, target_ms=45.0)
    assert trainer.count("over_budget") == 1


def test_one_slow_frame_resets_the_latency_streak():
    trainer = Trainer(sample_every=1)
    for frame in range(1, 4):
        trainer.observe(frame, frame_ms=90.0, target_ms=45.0)
    trainer.observe(4, frame_ms=40.0, target_ms=45.0)
    trainer.observe(5, frame_ms=90.0, target_ms=45.0)
    assert trainer.count("over_budget") == 0


# ─────────────────────────────── tuning ───────────────────────────────

def mismatch_trainer(**values):
    trainer = Trainer(sample_every=1, starting_values=values)
    observe(trainer, colour=31.9)
    return trainer


def test_a_colour_mismatch_moves_the_addressing_knob_one_step():
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    assert trainer.values["tone_transfer_strength"] == .6
    adjustment = trainer.adjustments[0]
    assert adjustment.knob == "tone_transfer_strength"
    assert adjustment.measured_before == 31.9
    assert adjustment.kept is None, "the change is not kept until later samples say so"


def test_a_change_that_improves_the_metric_is_kept():
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=8.0)
    assert trainer.adjustments[0].kept is True
    assert trainer.values["tone_transfer_strength"] == .6


def test_a_change_that_does_not_help_is_reverted():
    """The difference between a trainer and a guess: it checks."""
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=31.9)
    assert trainer.adjustments[0].kept is False
    assert trainer.values["tone_transfer_strength"] == .5, "the value was left changed"


def test_a_barely_better_change_is_still_reverted():
    """Below the margin the improvement is noise, and keeping it would be superstition."""
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=31.0)                     # 2.8% better
    assert trainer.adjustments[0].kept is False


def test_a_knob_that_fails_twice_is_locked_and_stops_being_tried():
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(12):
        observe(trainer, colour=31.9)
    assert "tone_transfer_strength" in trainer.locked
    assert len(trainer.adjustments) == 2, "it should stop after two reverts, not oscillate"


def test_a_reverted_change_is_not_immediately_tried_again():
    """Otherwise the session is spent half at a value already measured as not helping."""
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=31.9)                    # reverts at the third sample
    assert trainer.adjustments[0].kept is False
    assert trainer.values["tone_transfer_strength"] == .5

    observe(trainer, colour=31.9)
    assert len(trainer.adjustments) == 1, "it retried the change it had just reverted"


def test_only_one_change_is_under_verification_at_a_time():
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    observe(trainer, colour=31.9)
    assert len(trainer.adjustments) == 1


def test_tuning_stops_at_the_bound_and_records_why():
    trainer = mismatch_trainer(tone_transfer_strength=1.0)
    assert trainer.adjustments == []
    assert "tone_transfer_strength" in trainer.locked
    assert any("bound" in note for note in trainer.notes)


def test_a_bad_edge_moves_the_feather_knob():
    trainer = Trainer(sample_every=1, starting_values={"parser_feather": .035})
    observe(trainer, colour=4.0, seam=7.8)
    assert trainer.values["parser_feather"] == .045
    assert trainer.adjustments[0].knob == "parser_feather"


def test_a_feather_change_is_verified_against_the_seam():
    trainer = Trainer(sample_every=1, starting_values={"parser_feather": .035})
    observe(trainer, colour=4.0, seam=7.8)
    for _ in range(3):
        observe(trainer, colour=4.0, seam=1.2)
    assert trainer.adjustments[0].kept is True
    assert trainer.adjustments[0].objective == "seam"


def test_an_over_budget_session_reduces_the_most_expensive_optional_stage():
    trainer = Trainer(sample_every=1, starting_values={"restoration_visibility": .75},
                      active_stages={"restoration"})
    for frame in range(1, 13):
        trainer.observe(frame, frame_ms=90.0, target_ms=45.0)
    assert trainer.values["restoration_visibility"] < .75
    assert trainer.adjustments[0].knob == "restoration_visibility"


def test_a_stage_that_is_not_loaded_is_not_turned_down():
    """Restoration is off, so moving its strength costs nothing and proves nothing."""
    trainer = Trainer(sample_every=1, starting_values={"restoration_visibility": .75},
                      active_stages=set())
    for frame in range(1, 13):
        trainer.observe(frame, frame_ms=90.0, target_ms=45.0)
    assert trainer.adjustments == []
    assert trainer.values["restoration_visibility"] == .75
    assert any("no optional stage" in note for note in trainer.notes)


def test_over_budget_tuning_gives_up_once_restoration_is_already_off():
    trainer = Trainer(sample_every=1, starting_values={"restoration_visibility": 0.0},
                      active_stages={"restoration"})
    for frame in range(1, 13):
        trainer.observe(frame, frame_ms=90.0, target_ms=45.0)
    assert trainer.adjustments == []
    assert any("bound" in note for note in trainer.notes)


def test_the_trainer_never_writes_to_global_settings():
    """Sessions share one engine, so one session tuning itself must not tune the others."""
    from app.config import settings

    before = {name: getattr(settings, name) for name in KNOBS}
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(6):
        observe(trainer, colour=31.9, seam=7.8)
    assert trainer.values, "the session should have its own values"
    assert {name: getattr(settings, name) for name in KNOBS} == before


# ─────────────────────────────── repair ───────────────────────────────

def test_a_restoration_fault_switches_restoration_off_and_is_recorded():
    trainer = Trainer(sample_every=1, starting_values={"restoration_visibility": .75})
    trainer.notice_fault("restoration", "the model rejected an input frame")
    assert trainer.values["restoration_visibility"] == 0.0
    assert trainer.count("restoration_fault") == 1
    assert trainer.mistakes[0].severity == "critical"
    assert trainer.repairs[0].stage == "restoration"


def test_a_stage_that_fails_repeatedly_is_repaired_once():
    trainer = Trainer(sample_every=1)
    for _ in range(5):
        trainer.notice_fault("restoration", "same fault")
    assert len(trainer.repairs) == 1
    assert trainer.count("restoration_fault") == 1


# ─────────────────────────────── output ───────────────────────────────

def test_the_state_is_serialisable_for_the_studio_panel():
    """It crosses a WebRTC data channel as JSON, so numpy types would break it there and
    nowhere else."""
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    trainer.notice_fault("restoration", "bad model")
    payload = json.dumps(trainer.state())
    assert "tone_transfer_strength" in payload


def test_the_state_separates_what_it_kept_from_what_it_reverted():
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=8.0)
    state = trainer.state()
    assert state["kept"] == 1 and state["reverted"] == 0 and state["adjustments"] == 1


def test_the_report_names_the_defect_the_change_and_the_outcome():
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=8.0)
    report = trainer.report()
    assert "colour_mismatch" in report
    assert "tone_transfer_strength" in report
    assert "0.5 → 0.6" in report
    assert "31.9 → 8.0" in report


def test_a_reverted_change_shows_what_it_was_measured_against():
    """A verdict without the numbers is not checkable by the person reading it."""
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=31.9)
    report = trainer.report()
    assert "was reverted" in report
    assert "31.9 → 31.9" in report, report


def test_a_short_session_is_not_reported_as_zero_minutes():
    trainer = Trainer(sample_every=1)
    assert "0.0 minutes" not in trainer.report()
    assert "seconds" in trainer.report()


def test_a_defect_the_trainer_fixed_is_not_listed_as_unfixable():
    """Found by reading a generated report: it listed colour_mismatch under "what no
    setting can fix" while the measurements above it showed the fix."""
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(3):
        observe(trainer, colour=8.0)

    assert trainer.adjustments[0].kept is True
    assert trainer.count("colour_mismatch") == 1, "the defect should still be in the ledger"
    assert trainer._no_setting_fixed() == []

    report = trainer.report()
    assert "What no setting fixed" not in report


def test_a_defect_nothing_could_fix_is_listed_with_the_number_of_attempts():
    trainer = mismatch_trainer(tone_transfer_strength=.5)
    for _ in range(12):
        observe(trainer, colour=31.9)

    unfixed = trainer._no_setting_fixed()
    assert [mistake.kind for mistake, _ in unfixed] == ["colour_mismatch"]
    assert unfixed[0][1] >= 1, "a defect that was fought should say how often"

    report = trainer.report()
    assert "What no setting fixed" in report
    assert "reverted each one" in report


def test_a_defect_with_no_knob_says_so_rather_than_claiming_an_attempt():
    trainer = Trainer(sample_every=1)
    observe(trainer, colour=None, seam=None, mean_abs=0.0)      # nothing changed at all

    assert [mistake.kind for mistake, _ in trainer._no_setting_fixed()] == ["no_change"]
    report = trainer.report()
    assert "no setting addresses this" in report
    assert "reverted each one" not in report


def test_the_report_says_what_it_cannot_do():
    """The honesty boundary belongs in the artefact a person reads, not only in a README."""
    report = Trainer().report()
    assert "cannot retrain the swap model" in report
    assert "no swap weights" in report


def test_the_report_distinguishes_a_clean_session_from_an_unwatched_one():
    clean = Trainer(sample_every=1)
    observe(clean, colour=4.0, seam=1.2)
    assert "Nothing was detected" in clean.report()

    unwatched = Trainer(sample_every=1, enabled=False)
    assert "Watched 0 frames" in unwatched.report()


def test_a_disabled_trainer_measures_nothing_and_changes_nothing():
    trainer = Trainer(sample_every=1, enabled=False)
    assert trainer.should_sample(0) is False
    observe(trainer, colour=31.9)
    assert trainer.counts == {} and trainer.adjustments == []
    assert trainer.state()["monitoring"] is False


def test_the_report_is_written_once_and_where_it_was_asked_to(tmp_path):
    trainer = Trainer(sample_every=1, report_dir=tmp_path)
    first = trainer.write_report()
    assert first is not None and first.parent == tmp_path and first.is_file()
    assert "Session report" in first.read_text()
    assert trainer.write_report() is None, "the session must not be reported twice"


def test_a_session_id_cannot_write_outside_the_report_directory(tmp_path):
    trainer = Trainer(session_id="../../etc/passwd", sample_every=1, report_dir=tmp_path)
    path = trainer.write_report()
    assert path is not None
    assert path.parent == tmp_path
    assert "passwd" in path.name and ".." not in path.name


def test_no_report_directory_means_no_report():
    trainer = Trainer(sample_every=1, report_dir=None)
    assert trainer.write_report() is None


def test_sampling_skips_frames_that_are_already_over_budget():
    """Spending 5 ms measuring a frame that is already late makes the lateness worse."""
    trainer = Trainer(sample_every=1)
    assert trainer.should_sample(0, was_over_budget=True) is False
    assert trainer.skipped_for_budget == 1
    assert trainer.should_sample(0, was_over_budget=False) is True


def test_sampling_happens_every_n_frames_not_every_frame():
    trainer = Trainer(sample_every=30)
    sampled = [frame for frame in range(90) if trainer.should_sample(frame)]
    assert sampled == [0, 30, 60], f"expected a sample every 30 frames, got {sampled}"


def test_the_summary_line_separates_a_quiet_session_from_a_loud_one():
    quiet = Trainer(sample_every=1)
    observe(quiet, colour=4.0, seam=1.2)
    assert "0 defects" in quiet.summary_line()

    loud = mismatch_trainer(tone_transfer_strength=.5)
    assert "1 defects" in loud.summary_line() or "defects" in loud.summary_line()


# ─────────────────────────────── knobs ───────────────────────────────

def test_a_knob_never_leaves_its_bounds():
    knob = KNOBS["restoration_visibility"]
    assert knob.clamp(5) == 1.0
    assert knob.clamp(-3) == 0.0
    assert knob.clamp(.55) == .55


def test_a_knob_offers_the_nearest_options_first():
    knob = KNOBS["parser_feather"]
    options = knob.candidates(.04)
    assert options
    assert all(knob.low <= value <= knob.high for value in options)
    assert options[0] in (.05, .03)


def test_a_knob_cannot_be_pushed_past_its_bound():
    """The bound is directional: at the top it can still come down, and it cannot go up."""
    assert KNOBS["tone_transfer_strength"].step_towards(1.0, 1) is None
    assert KNOBS["tone_transfer_strength"].step_towards(1.0, -1) == .9
    assert KNOBS["tone_transfer_strength"].step_towards(0.0, -1) is None
    assert KNOBS["tone_transfer_strength"].step_towards(0.0, 1) == .1


def test_settings_that_must_not_be_tuned_are_recorded_as_such():
    for name in ("temporal_strength", "verification_threshold", "max_frame_width"):
        assert name not in KNOBS
        assert name in FIXED


def test_a_hand_edited_tuned_file_cannot_escape_the_bounds(tmp_path):
    """The file is data on disk: it gets clamped, not trusted."""
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"values": {"restoration_visibility": 40,
                                           "tone_transfer_strength": -9,
                                           "parser_feather": .05}}))
    tuned = read_tuned(path)
    assert tuned["restoration_visibility"] == 1.0
    assert tuned["tone_transfer_strength"] == 0.0
    assert tuned["parser_feather"] == .05


def test_a_tuned_file_cannot_add_settings_that_do_not_exist(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"values": {"verification_threshold": 0.0,
                                           "definitely_not_a_setting": 1}}))
    assert read_tuned(path) == {}


def test_a_broken_tuned_file_is_ignored_rather_than_raising(tmp_path):
    for content in ("", "{not json", "[]", json.dumps({"values": []})):
        path = tmp_path / "tuned.json"
        path.write_text(content)
        assert read_tuned(path) == {}


def test_a_missing_tuned_file_is_not_an_error(tmp_path):
    assert read_tuned(tmp_path / "absent.json") == {}


def test_applying_tuned_values_clamps_and_reports_what_it_applied(tmp_path):
    from app.config import Settings

    settings = Settings()
    applied = apply_to_settings(settings, {"restoration_visibility": 40,
                                           "unknown_name": 1}, source="test")
    assert applied == {"restoration_visibility": 1.0}
    assert settings.restoration_visibility == 1.0


def test_applying_tuned_values_ignores_a_non_number():
    from app.config import Settings

    settings = Settings()
    assert apply_to_settings(settings, {"parser_feather": "wide"}) == {}
    assert settings.parser_feather == .035


# ─────────────────── feeding the offline search from a live session ───────────────────

def test_a_saved_frame_is_a_pair_the_offline_search_can_read(tmp_path):
    """The loop the feature depends on: a live session's own failure becomes data.

    The filename is the contract, not a detail — `pairs_from_directory` pairs
    `X_original.*` with `X_swapped.*`, so a near-miss here looks like a directory of
    unusable images and the search reports "nothing found", which reads like success.
    Asserted as a round trip through the reader rather than as a file existing.
    """
    from tools.train_defaults import pairs_from_directory

    trainer = Trainer(sample_every=1, capture_dir=tmp_path, capture_limit=4)
    original = np.full((60, 80, 3), 100, np.uint8)
    swapped = np.full((60, 80, 3), 160, np.uint8)

    written = trainer.save_pair(original, swapped, 7)

    assert written is not None and written.exists()
    pairs = pairs_from_directory(tmp_path)
    assert len(pairs) == 1, f"the reader did not pair the files: {list(tmp_path.iterdir())}"
    assert np.array_equal(pairs[0].original, original), "channels came back swapped"
    assert np.array_equal(pairs[0].swapped, swapped)
    assert trainer.captures == 1
    assert trainer.state()["captures"] == 1


def test_saving_stops_at_the_cap(tmp_path):
    """A session that watches thousands of frames must not fill the disk."""
    trainer = Trainer(sample_every=1, capture_dir=tmp_path, capture_limit=2)

    for frame in range(5):
        trainer.save_pair(np.full((8, 8, 3), 10, np.uint8), np.full((8, 8, 3), 20, np.uint8), frame)

    assert trainer.captures == 2
    assert len(list(tmp_path.glob("*_original.png"))) == 2


def test_nothing_is_saved_unless_it_was_asked_for(tmp_path):
    """Off by default: two PNG encodes cost more than measuring the frame."""
    trainer = Trainer(sample_every=1, capture_dir=tmp_path, capture_limit=0)

    assert trainer.capture_dir is None
    assert trainer.save_pair(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), 1) is None
    assert list(tmp_path.iterdir()) == []

    disabled = Trainer(sample_every=1, enabled=False, capture_dir=tmp_path, capture_limit=4)
    assert disabled.save_pair(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), 1) is None


def test_the_report_says_what_was_saved_and_where(tmp_path):
    trainer = Trainer(sample_every=1, capture_dir=tmp_path, capture_limit=4)
    trainer.save_pair(np.full((8, 8, 3), 10, np.uint8), np.full((8, 8, 3), 20, np.uint8), 1)

    report = trainer.report()
    assert "Saved 1 measured frames" in report
    assert "--pairs" in report, "the report should say what to do with them"
