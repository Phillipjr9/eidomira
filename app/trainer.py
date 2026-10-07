"""The trainer: it watches every frame, records what went wrong, and tunes what it may.

What this is, and what it is not.

It **cannot retrain the swap model**. There are no weights in `models/`, no training data
and no accelerator in the loop, so anything claiming to improve the neural network would be
inventing numbers. What it does instead is real:

* **Monitor.** Every sampled frame is measured with the same metrics
  (`tools/quality_report.py`) the project already trusts, and every defect is written to a
  ledger with the numbers that produced it. The three silent defects fixed in the
  compositor were all of the class this catches: a stage that fails, reports success, and
  leaves the frame untouched.
* **Tune, within bounds.** When a measured defect has a knob that addresses it, the trainer
  moves that knob one step and *verifies* the change against later samples, reverting it if
  the metric did not improve. A knob reverted twice is locked for the session, because a
  controller that keeps trying the same failed change is an oscillator, not a learner.
* **Repair.** A stage that reports a fault is switched off rather than run on every frame,
  and the fault is recorded as a defect instead of scrolling past in a log.
* **Report.** At the end of a session it writes a plain-language summary: what was wrong,
  what it changed, what it tried and reverted, and what a person should fix in the code
  because no setting can.

The one thing it may not do is move a setting nobody measured. `restoration_visibility` is
skipped when no restoration model is installed, and the offline trainer refuses to score a
knob it cannot evaluate on the frames it was given, rather than inventing a good value.

Cost is the reason for `sample_every`. A full-resolution quality sample costs **64 ms** at
960x738 on a 2-vCPU box -- more than the entire 45 ms frame budget -- so measurement happens
on a frame downscaled to `MEASURE_WIDTH` (~5 ms) and only every `sample_every` frames, and
it is skipped entirely on frames that were already over budget.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.knobs import KNOBS

#: Where measurement happens. Thresholds below are calibrated at this width, so a change
#: here invalidates them and the calibration test is what says so.
MEASURE_WIDTH = 256

#: Calibrated on synthetic pairs at MEASURE_WIDTH (see tests/test_trainer.py): a healthy
#: feathered composite reads a seam of about 1.0-1.7 and a hard cut around 7.8, so 3.0 sits
#: between "soft edge" and "visible line". A matched face reads colour 0-6; a real lighting
#: mismatch reads 12-33, so 10.0 sits between them.
SEAM_LIMIT = 3.0
COLOUR_LIMIT = 10.0

#: Below this mean absolute difference, a frame that claimed to have found a face is
#: indistinguishable from the frame that went in, which is the silent no-op signature.
NO_CHANGE_LIMIT = 0.5

#: Latency is judged against the preset's own target, with a tolerance so that ordinary
#: jitter is not a defect, and only after a run of frames rather than on one slow frame.
LATENCY_TOLERANCE = 1.35
OVER_BUDGET_STREAK = 4

#: A change is kept only if the metric improves by this much; it is reverted otherwise.
IMPROVE_MARGIN = 0.05
VERIFY_SAMPLES = 3

#: A knob that was just reverted gets this many samples of rest before it is tried again.
#: Without it the trainer re-applies the change on the very next sample, which spends half
#: the session at a value it has already measured as not helping.
COOLDOWN_SAMPLES = 6
LOCK_AFTER_REVERTS = 2
MAX_ADJUSTMENTS = 40

#: Which knob objective addresses which defect, so the report can say honestly whether a
#: defect was left alone or fought and won. A defect with no objective has no knob at all.
KIND_OBJECTIVES = {
    "colour_mismatch": "colour",
    "visible_edge": "seam",
    "over_budget": "frame_ms",
}


@dataclass(frozen=True)
class Mistake:
    """One recorded defect, with the measurement that produced it."""

    kind: str
    summary: str
    severity: str
    frame: int
    evidence: dict = field(default_factory=dict)


@dataclass
class Adjustment:
    """One bounded change, and whether measuring later said it helped."""

    knob: str
    before: float
    after: float
    reason: str
    objective: str
    frame: int
    measured_before: float | None = None
    measured_after: float | None = None
    kept: bool | None = None      # None while the change is still being verified

    @property
    def improved_by(self) -> float | None:
        if self.measured_before in (None, 0) or self.measured_after is None:
            return None
        return 1.0 - (self.measured_after / self.measured_before)


@dataclass
class Repair:
    """A stage switched off, and why."""

    stage: str
    detail: str
    action: str


class Trainer:
    """Per-session supervisor. One instance per stream, never shared."""

    def __init__(self, session_id: str = "session", *, sample_every: int = 30,
                 measure_width: int = MEASURE_WIDTH, enabled: bool = True,
                 report_dir: Path | str | None = "reports",
                 starting_values: dict | None = None,
                 active_stages: set[str] | None = None):
        self.session_id = str(session_id)
        self.enabled = bool(enabled)
        self.sample_every = max(1, int(sample_every))
        self.measure_width = max(64, int(measure_width))
        self.report_dir = Path(report_dir) if report_dir else None
        #: Stages the engine is really running. Turning down a stage that is not loaded is
        #: a change that costs nothing and proves nothing, and it would score as an
        #: improvement on the next sample for no reason at all.
        self.active_stages = set(active_stages or ())

        #: Knob values this session is running with. Passed to the engine per frame, never
        #: written to `settings`, because the engine is shared by every live session and
        #: one session's tuning must not change another's output.
        self.values: dict[str, float] = dict(starting_values or {})

        self.mistakes: list[Mistake] = []
        #: How often each defect kind was seen. The ledger holds one entry per kind so it
        #: stays readable, and this is what keeps the count honest.
        self.counts: Counter = Counter()
        self.adjustments: list[Adjustment] = []
        self.repairs: list[Repair] = []
        self.notes: list[str] = []

        self.frames = 0
        self.samples = 0
        self.skipped_for_budget = 0
        self.started = time.monotonic()

        self._last_metrics: dict | None = None
        self._over_budget = 0
        self._pending: Adjustment | None = None
        self._verifying: list[float] = []
        self._reverts: Counter = Counter()
        self._cooldown: Counter = Counter()
        self._locked: set[str] = set()
        self._reported = False

    # ───────────────────────────── measurement ─────────────────────────────

    @property
    def overrides(self) -> dict:
        """Values to hand the engine for this frame. Empty when nothing was tuned."""
        return dict(self.values)

    def should_sample(self, frame: int, was_over_budget: bool = False) -> bool:
        """Whether this frame is a candidate for measurement.

        Sampling is skipped on frames that were already over budget: the trainer's job is
        to make the frame cheaper, and spending 5 ms of an overrun frame measuring it makes
        the overrun worse.
        """
        if not self.enabled:
            return False
        if was_over_budget:
            self.skipped_for_budget += 1
            return False
        return int(frame) % self.sample_every == 0

    def measure(self, original_rgb: np.ndarray, result_rgb: np.ndarray) -> dict | None:
        """Score one frame pair, downscaled to the calibrated width.

        Returns None when the pair cannot be compared at all (shape mismatch, or nothing in
        the frame changed), because a metric over nothing is not a zero.
        """
        if original_rgb.shape != result_rgb.shape:
            return None
        small = []
        for frame in (original_rgb, result_rgb):
            scale = self.measure_width / frame.shape[1]
            if scale < 1.0:
                size = (self.measure_width, max(2, int(frame.shape[0] * scale)))
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            small.append(frame)
        original, result = small

        from tools.quality_report import colour_shift, seam_ratio

        delta = np.abs(original.astype(np.int16) - result.astype(np.int16))
        metrics = {
            "mean_abs": round(float(delta.mean()), 4),
            "seam": seam_ratio(original, result),
            "colour": (colour_shift(original, result) or {}).get("distance"),
        }
        return metrics

    # ───────────────────────────── observation ─────────────────────────────

    def observe(self, frame: int, *, frame_ms: float | None = None,
                target_ms: float | None = None, face_found: bool = True,
                metrics: dict | None = None, steady_state: bool = True) -> None:
        """Take in one frame's numbers and act on them."""
        self.frames = max(self.frames, int(frame))
        if not self.enabled:
            return

        if metrics is not None:
            self._last_metrics = metrics
            self.samples += 1
            self._detect_quality(metrics, face_found, steady_state, frame)
            self._verify(metrics, frame)
            for name in list(self._cooldown):
                if self._cooldown[name] > 0:
                    self._cooldown[name] -= 1

        if frame_ms is not None and target_ms:
            self._detect_latency(frame_ms, target_ms, frame)

    def notice_fault(self, stage: str, detail: str) -> None:
        """A stage reported a fault. Record it and take the stage out of the loop."""
        if not self.enabled or any(r.stage == stage for r in self.repairs):
            return
        self._record(f"{stage}_fault", "critical",
                     f"the {stage} stage failed and was switched off: {detail}",
                     self.frames, {"detail": detail})
        action = "reported in the session summary"
        if stage == "restoration" and "restoration_visibility" in KNOBS:
            self._set("restoration_visibility", 0.0)
            action = "restoration disabled for this session"
        self.repairs.append(Repair(stage=stage, detail=detail, action=action))

    # ───────────────────────────── detection ─────────────────────────────

    def _detect_quality(self, metrics: dict, face_found: bool, steady_state: bool,
                        frame: int) -> None:
        colour, seam = metrics.get("colour"), metrics.get("seam")

        if face_found and steady_state and metrics.get("mean_abs", 1.0) < NO_CHANGE_LIMIT:
            self._record("no_change", "warning",
                         "a face was found but the output is unchanged from the input: "
                         "the swap did nothing", frame, metrics)
            return

        if colour is not None and colour > COLOUR_LIMIT:
            self._record("colour_mismatch", "warning",
                         f"the swapped face is {colour:.0f} levels from the colours around "
                         "it, which reads as a face pasted from another room",
                         frame, metrics)
            self._tune("tone_transfer_strength", 1, "colour", frame,
                       f"colour mismatch measured at {colour:.1f}")
        elif seam is not None and seam > SEAM_LIMIT:
            self._record("visible_edge", "warning",
                         f"the edge of the composited region is {seam:.1f}x sharper than "
                         "the texture it sits in", frame, metrics)
            self._tune("parser_feather", 1, "seam", frame,
                       f"edge measured at {seam:.1f}")

    def _detect_latency(self, frame_ms: float, target_ms: float, frame: int) -> None:
        if frame_ms > target_ms * LATENCY_TOLERANCE:
            self._over_budget += 1
        else:
            self._over_budget = 0
        if self._over_budget < OVER_BUDGET_STREAK:
            return
        self._over_budget = 0
        self._record("over_budget", "warning",
                     f"the last {OVER_BUDGET_STREAK} frames cost more than "
                     f"{LATENCY_TOLERANCE:.2f}x the {target_ms:.0f} ms budget",
                     frame, {"frame_ms": round(frame_ms, 1), "target_ms": target_ms})
        # Restoration is the most expensive optional stage, so it is what gives first.
        if "restoration" in self.active_stages:
            self._tune("restoration_visibility", -1, "frame_ms", frame,
                       f"frames over budget at {frame_ms:.1f} ms", sample=frame_ms)
        else:
            self.notes.append(
                f"frames are over budget at {frame_ms:.1f} ms but no optional stage is "
                "loaded to turn down; resolution belongs to the adaptive controller")

    # ───────────────────────────── tuning ─────────────────────────────

    def _record(self, kind: str, severity: str, summary: str, frame: int,
                evidence: dict) -> None:
        # Appending every occurrence would bury the ledger under a defect that repeats on
        # every sample, so the first one is kept with its evidence and the rest are counted.
        self.counts[kind] += 1
        if self.counts[kind] == 1:
            self.mistakes.append(Mistake(kind=kind, severity=severity, summary=summary,
                                         frame=frame, evidence=dict(evidence)))

    def count(self, kind: str) -> int:
        """How many times a defect was seen, including the repeats the ledger collapses."""
        return int(self.counts.get(kind, 0))

    def _tune(self, knob_name: str, direction: int, objective: str, frame: int,
              reason: str, sample: float | None = None) -> None:
        if self._pending is not None:
            return                                   # one change under verification at a time
        if knob_name in self._locked or self._cooldown[knob_name] > 0:
            return
        if len(self.adjustments) >= MAX_ADJUSTMENTS:
            self.notes.append(
                f"stopped tuning after {MAX_ADJUSTMENTS} changes; the remaining defects are "
                "in the summary instead of being papered over with settings")
            self._locked.add(knob_name)
            return
        knob = KNOBS.get(knob_name)
        if knob is None:
            return

        current = self.values.get(knob_name)
        if current is None:
            from app.config import settings
            current = float(getattr(settings, knob_name))
        after = knob.step_towards(current, direction)
        if after is None:
            self.notes.append(
                f"{knob_name} is already at its {'upper' if direction > 0 else 'lower'} "
                f"bound ({knob.clamp(current)}); {reason} cannot be fixed by moving it")
            self._locked.add(knob_name)
            return

        measured = sample if sample is not None else self._objective_value(objective)
        self._set(knob_name, after)
        self._pending = Adjustment(knob=knob_name, before=knob.clamp(current), after=after,
                                   reason=reason, objective=objective, frame=frame,
                                   measured_before=measured)
        self._verifying = []
        self.adjustments.append(self._pending)

    def _objective_value(self, objective: str) -> float | None:
        if not self._last_metrics:
            return None
        return self._last_metrics.get(objective)

    def _set(self, knob_name: str, value: float) -> None:
        self.values[knob_name] = KNOBS[knob_name].clamp(value)

    def _verify(self, metrics: dict, frame: int) -> None:
        """Decide whether the change under verification actually helped."""
        pending = self._pending
        if pending is None:
            return
        observed = metrics.get(pending.objective)
        if observed is None:
            return
        self._verifying.append(float(observed))
        if len(self._verifying) < VERIFY_SAMPLES:
            return

        after = float(np.median(self._verifying))
        pending.measured_after = round(after, 4)
        before = pending.measured_before
        improved = (before is not None and before > 0
                    and after < before * (1.0 - IMPROVE_MARGIN))
        if improved:
            pending.kept = True
        else:
            pending.kept = False
            self._set(pending.knob, pending.before)
            self._reverts[pending.knob] += 1
            self._cooldown[pending.knob] = COOLDOWN_SAMPLES
            if self._reverts[pending.knob] >= LOCK_AFTER_REVERTS:
                self._locked.add(pending.knob)
        self._pending = None
        self._verifying = []

    @property
    def locked(self) -> list[str]:
        return sorted(self._locked)

    # ───────────────────────────── output ─────────────────────────────

    def state(self) -> dict:
        """What the studio panel shows, and what the report is built from."""
        return {
            "monitoring": self.enabled,
            "samples": self.samples,
            "mistakes": sum(self.counts.values()),
            "kinds": len(self.mistakes),
            "counts": dict(self.counts),
            "adjustments": len(self.adjustments),
            "kept": sum(1 for a in self.adjustments if a.kept is True),
            "reverted": sum(1 for a in self.adjustments if a.kept is False),
            "pending": self._pending.knob if self._pending else None,
            "locked": self.locked,
            "repairs": [asdict(r) for r in self.repairs],
            "values": {k: round(v, 3) for k, v in self.values.items()},
            "recent": [
                {"kind": m.kind, "summary": m.summary, "severity": m.severity}
                for m in self.mistakes[-4:]
            ],
            "notes": self.notes[-3:],
        }

    def report(self) -> str:
        """A plain-language summary of the session."""
        seconds = max(0.0, time.monotonic() - self.started)
        duration = (f"{seconds:.0f} seconds" if seconds < 60
                    else f"{seconds / 60.0:.1f} minutes")
        lines = [
            f"# Session report — {self.session_id}",
            "",
            f"Watched {self.frames} frames over {duration} and measured "
            f"{self.samples} of them.",
        ]
        if self.skipped_for_budget:
            lines.append(
                f"Skipped {self.skipped_for_budget} measurements because the frame was "
                "already over budget."
            )
        lines.append("")

        if not self.mistakes:
            lines += ["## What went wrong", "", "Nothing was detected.", ""]
        else:
            lines += ["## What went wrong", ""]
            for mistake in self.mistakes:
                seen = self.counts[mistake.kind]
                times = "once" if seen == 1 else f"{seen}x"
                lines.append(f"- **{mistake.kind}** ({mistake.severity}, first at frame "
                             f"{mistake.frame}, {times}) — {mistake.summary}")
            lines.append("")

        if self.repairs:
            lines += ["## What it repaired", ""]
            for repair in self.repairs:
                lines.append(f"- **{repair.stage}** — {repair.detail}. {repair.action}.")
            lines.append("")

        kept = [a for a in self.adjustments if a.kept is True]
        reverted = [a for a in self.adjustments if a.kept is False]
        pending = [a for a in self.adjustments if a.kept is None]

        lines += ["## What it changed", ""]
        if not kept:
            lines.append("Nothing was changed and kept." if self.adjustments
                         else "Nothing needed changing.")
        for adjustment in kept:
            lines.append(
                f"- `{adjustment.knob}` {adjustment.before} → {adjustment.after} "
                f"({adjustment.objective} "
                f"{adjustment.measured_before:.1f} → {adjustment.measured_after:.1f}) "
                f"because {adjustment.reason}"
            )
        lines.append("")

        if reverted or pending:
            lines += ["## What it tried and undid", ""]
            for adjustment in reverted:
                improvement = adjustment.improved_by
                verdict = ("no improvement" if improvement is None or improvement <= 0
                           else f"only {improvement:.0%} better, below the "
                                f"{IMPROVE_MARGIN:.0%} needed")
                measured = (
                    f" ({adjustment.objective} {adjustment.measured_before:.1f} → "
                    f"{adjustment.measured_after:.1f})"
                    if adjustment.measured_before is not None
                    and adjustment.measured_after is not None else ""
                )
                lines.append(
                    f"- `{adjustment.knob}` {adjustment.before} → {adjustment.after} was "
                    f"reverted{measured}: {verdict}"
                )
            for adjustment in pending:
                lines.append(
                    f"- `{adjustment.knob}` {adjustment.before} → {adjustment.after} was "
                    "still being verified when the session ended"
                )
            lines.append("")

        if self.locked:
            lines += [
                "## Settings locked for the session", "",
                "These were moved and reverted, or are already at their bound, so the "
                "trainer stopped trying rather than oscillate:", "",
            ]
            for name in self.locked:
                lines.append(f"- `{name}` — {KNOBS[name].why}" if name in KNOBS
                             else f"- `{name}`")
            lines.append("")

        unfixed = self._no_setting_fixed()
        if unfixed:
            lines += ["## What no setting fixed", "",
                      "Reported, and either nothing here addresses them or every change the "
                      "trainer tried was reverted:", ""]
            for mistake, attempts in unfixed:
                if attempts:
                    tail = (f"tried {attempts} setting change(s) and reverted each one, so "
                            "this needs a change in the code or a different model")
                else:
                    tail = ("no setting addresses this, so it needs a change in the code or "
                            "a different model")
                lines.append(f"- {mistake.kind}: {mistake.summary} — {tail}")
            lines += [
                "",
                "A defect the trainer fixed is deliberately not listed here: it belongs in "
                "the section above, with the measurement that shows the fix.",
                "",
            ]

        lines += [
            "## What this report is not", "",
            "The trainer tunes settings. It cannot retrain the swap model: there are no "
            "swap weights in `models/` and no accelerator in this loop, so nothing here "
            "improves the network itself.",
        ]
        return "\n".join(lines)

    def _no_setting_fixed(self) -> list[tuple[Mistake, int]]:
        """Defects that survived the session, with how many changes were tried against them.

        A defect is only "unfixed" if no change aimed at it was kept. Listing one that the
        trainer just measured a fix for would be the report contradicting itself.
        """
        survivors = []
        for mistake in self.mistakes:
            objective = KIND_OBJECTIVES.get(mistake.kind)
            attempts = [a for a in self.adjustments if objective and a.objective == objective]
            if objective is None or not any(a.kept for a in attempts):
                survivors.append((mistake, len(attempts)))
        return survivors

    def write_report(self, directory: Path | str | None = None) -> Path | None:
        """Write the report once. Returns the path, or None when there is nowhere to write."""
        if self._reported or not self.enabled:
            return None
        target_dir = Path(directory) if directory else self.report_dir
        if target_dir is None:
            return None
        self._reported = True
        safe = "".join(c for c in self.session_id if c.isalnum() or c in "-_")[:48] or "session"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{safe}-{stamp}.md"
        path.write_text(self.report())
        return path

    def summary_line(self) -> str:
        """One line for a log, so a session that found nothing is distinguishable."""
        return (f"trainer: {self.samples} samples, {sum(self.counts.values())} defects, "
                f"{sum(1 for a in self.adjustments if a.kept)} settings kept, "
                f"{len(self.repairs)} stages repaired")
