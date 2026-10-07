from __future__ import annotations

from dataclasses import dataclass, field
import random
import time


@dataclass
class LivenessChallenge:
    """Short randomized pose sequence; session-local and time limited."""
    steps: list[str] = field(default_factory=list)
    index: int = 0
    started: float = field(default_factory=time.monotonic)
    expires_seconds: float = 25.0
    hold_frames: int = 0
    required_hold_frames: int = 3

    def __post_init__(self):
        if not self.steps:
            sides = ["side_a", "side_b"]
            random.SystemRandom().shuffle(sides)
            self.steps = ["center", *sides, "center"]

    @property
    def complete(self):
        return self.index >= len(self.steps)

    @property
    def expired(self):
        return time.monotonic() - self.started > self.expires_seconds

    @property
    def current(self):
        return "complete" if self.complete else self.steps[self.index]

    @property
    def instruction(self):
        return {
            "center": "Look directly at the camera",
            "side_a": "Slowly turn your head to one side",
            "side_b": "Now turn your head to the other side",
            "complete": "Live presence confirmed",
        }[self.current]

    def observe(self, yaw: float | None, face_found: bool):
        if self.expired:
            return {"complete": False, "expired": True, "instruction": "Challenge expired. Restart the session."}
        if not face_found or yaw is None:
            self.hold_frames = 0
            return self.payload("Keep one face clearly visible")
        matches = {
            "center": abs(yaw) < .10,
            "side_a": yaw < -.16,
            "side_b": yaw > .16,
        }.get(self.current, False)
        self.hold_frames = self.hold_frames + 1 if matches else 0
        if self.hold_frames >= self.required_hold_frames:
            self.index += 1
            self.hold_frames = 0
        return self.payload()

    def payload(self, override: str | None = None):
        return {"complete": self.complete, "expired": self.expired,
                "step": min(self.index + 1, len(self.steps)), "total": len(self.steps),
                "instruction": override or self.instruction,
                "progress": round(self.index / len(self.steps), 2)}
