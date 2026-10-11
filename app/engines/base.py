from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Enrollment:
    identity: object
    preview: np.ndarray


@dataclass
class FrameResult:
    image: np.ndarray
    face_found: bool
    verified: bool
    latency_ms: float


class FaceSwapEngine(ABC):
    name = "base"

    @abstractmethod
    def enroll(self, rgb: np.ndarray) -> Enrollment: ...

    @abstractmethod
    def process(self, rgb: np.ndarray, identity: object, verified: bool,
                overrides: dict | None = None) -> FrameResult:
        """One frame.

        `overrides` carries per-session quality values from the trainer. They arrive per
        call rather than being written to the engine because the engine is shared by every
        live session: one session tuning itself must not change what another one sees.
        """
        ...

    def stage_faults(self) -> dict[str, str]:
        """Optional stages that have failed, by name, for the trainer to act on."""
        return {}

    def active_stages(self) -> set[str]:
        """Optional stages this engine is really running.

        The trainer will not turn down a stage that is not loaded — that would be a change
        that costs nothing and proves nothing.
        """
        return set()

    @abstractmethod
    def verify_self(self, rgb: np.ndarray, identity: object) -> tuple[bool, float]: ...

    @abstractmethod
    def observe_liveness(self, rgb: np.ndarray) -> tuple[bool, float | None]:
        """Return (face_found, normalized yaw proxy)."""
        ...
