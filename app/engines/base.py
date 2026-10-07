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
    def process(self, rgb: np.ndarray, identity: object, verified: bool) -> FrameResult: ...

    @abstractmethod
    def verify_self(self, rgb: np.ndarray, identity: object) -> tuple[bool, float]: ...

    @abstractmethod
    def observe_liveness(self, rgb: np.ndarray) -> tuple[bool, float | None]:
        """Return (face_found, normalized yaw proxy)."""
        ...
