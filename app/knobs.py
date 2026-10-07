"""Which quality settings may be tuned automatically, and how far they may move.

The live trainer (`app/trainer.py`) and the offline trainer (`tools/train_defaults.py`) both
read this table, so a value one of them considers safe is a value the other is allowed to
write. A tuned value that the live loop clamps at one bound and the offline search writes
at another would be worse than having no bounds at all.

The bounds are here rather than at each call site because an unbounded search for "better"
will always find a setting that scores well on the metric and looks wrong to a person:

* restoration at full strength maximises every sharpness measure and produces a plastic
  face, which is why the recommended range in the restoration literature is 0.7-0.8;
* feathering past its upper bound removes the seam by removing the mask;
* tone transfer past 1.0 stops correcting the swapped face toward the target and starts
  replacing it with the target, which defeats the swap.

This module deliberately imports nothing from the application, so `app.config` can read
tuned values at import time without a circular import.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("eidomira.knobs")

#: Where the offline trainer records what it measured. Absent by default: nothing is tuned
#: until `tools/train_defaults.py` has run and had something to measure.
TUNED_DEFAULTS_PATH = Path("data/tuned_defaults.json")


@dataclass(frozen=True)
class Knob:
    """One setting, with the range it is safe to move within and why."""

    name: str
    low: float
    high: float
    step: float
    why: str

    def clamp(self, value) -> float:
        """Force a value into range. A wrong type is a bug, not a tuning decision."""
        return round(min(self.high, max(self.low, float(value))), 4)

    def in_range(self, value) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return self.low <= number <= self.high

    def step_towards(self, value: float, direction: int) -> float | None:
        """One step from `value` in `direction`, or None when it cannot move that way.

        This is the question the live trainer asks: a knob already at its upper bound
        cannot fix a defect by going up, and saying so is more useful than returning the
        bound and pretending something changed.
        """
        current = self.clamp(value)
        moved = self.clamp(value + (1 if direction >= 0 else -1) * self.step)
        return None if moved == current else moved

    def candidates(self, value: float) -> list[float]:
        """Values to try from here: one step each way, nearest first, inside the bounds.

        Ordered by how far they move the knob, so a search that is cut short has still
        tried the smallest change that might help.
        """
        options = []
        for direction in (1, -1):
            candidate = self.clamp(value + direction * self.step)
            if candidate != self.clamp(value):
                options.append(candidate)
        return options


#: The tunable surface. Everything else in the pipeline is either fixed by physics
#: (frame size, orientation) or already owned by the adaptive controller
#: (`temporal_strength` is derived from the preset in `app/adaptive.py`, and a second
#: controller moving it would fight the first).
KNOBS: dict[str, Knob] = {
    "tone_transfer_strength": Knob(
        "tone_transfer_strength", 0.0, 1.0, .1,
        "past 1.0 the correction stops matching the target's lighting and starts "
        "replacing the swapped face with it",
    ),
    "restoration_visibility": Knob(
        "restoration_visibility", 0.0, 1.0, .1,
        "full strength is reported to look airbrushed; 0.7-0.8 is the documented range",
    ),
    "parser_feather": Knob(
        "parser_feather", .01, .08, .01,
        "below the lower bound the composited edge is visible as a line; above the upper "
        "bound the mask is so soft it stops protecting hair and glasses",
    ),
}

#: Knobs the trainer will never move, with the reason, so that "it did not touch this" is
#: a decision on record rather than an oversight.
FIXED = {
    "max_frame_width": "the adaptive controller owns resolution; two controllers fighting "
                       "produced the oscillation this table exists to prevent",
    "temporal_strength": "derived from the quality preset in app/adaptive.py",
    "verification_threshold": "an identity decision, not a quality preference: moving it "
                              "silently changes who is allowed through",
    "max_active_peers": "capacity, not quality",
}


def apply_to_settings(settings, values: dict, *, source: str = "tuned defaults") -> dict:
    """Set knobs on a settings object, clamped, returning what was actually applied.

    Unknown names are ignored rather than set, because a typo in a data file must not be
    able to add attributes to the settings object that nothing reads.
    """
    applied = {}
    for name, value in (values or {}).items():
        knob = KNOBS.get(name)
        if knob is None:
            log.warning("ignoring %r from %s: not a tunable setting", name, source)
            continue
        try:
            wanted = knob.clamp(value)
        except (TypeError, ValueError):
            log.warning("ignoring %r=%r from %s: not a number", name, value, source)
            continue
        current = getattr(settings, name, None)
        if current is None:
            log.warning("ignoring %r from %s: no such setting", name, source)
            continue
        if wanted != current:
            setattr(settings, name, wanted)
        applied[name] = wanted
    return applied


def read_tuned(path: Path | str = TUNED_DEFAULTS_PATH) -> dict:
    """Load measured values, clamped, ignoring anything malformed.

    Treats the file as untrusted input: it is written by a tool, may be hand-edited, and
    may be left over from a version with different bounds. Every value goes through
    `Knob.clamp`, so a file saying `restoration_visibility: 40` cannot put the pipeline
    somewhere the bounds were designed to prevent.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s (%s); using defaults", path, exc)
        return {}
    if not isinstance(payload, dict):
        log.warning("%s is not an object; using defaults", path)
        return {}

    values = payload.get("values", payload)
    if not isinstance(values, dict):
        log.warning("%s has no usable 'values' object; using defaults", path)
        return {}

    tuned = {}
    for name, knob in KNOBS.items():
        if name not in values:
            continue
        try:
            tuned[name] = knob.clamp(values[name])
        except (TypeError, ValueError):
            log.warning("ignoring %r=%r in %s: not a number", name, values[name], path)
    return tuned
