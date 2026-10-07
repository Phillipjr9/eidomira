from __future__ import annotations
import time
import cv2
import numpy as np
from .base import Enrollment, FaceSwapEngine, FrameResult


class DiagnosticEngine(FaceSwapEngine):
    """Honest transport/UI test backend. It never pretends to be neural inference."""
    name = "diagnostic"
    providers = ()
    provider = None
    accelerated = False

    def enroll(self, rgb: np.ndarray) -> Enrollment:
        if rgb is None or min(rgb.shape[:2]) < 64:
            raise ValueError("Use a clear portrait at least 64×64 pixels.")
        return Enrollment(identity={"authorized": True}, preview=rgb)

    def verify_self(self, rgb, identity):
        return True, 1.0

    def observe_liveness(self, rgb):
        # Cycles the challenge in diagnostic mode so UI/transport can be tested.
        count = getattr(self, "_diagnostic_pose_count", 0)
        sequence = [0.0] * 3 + [-.3] * 3 + [.3] * 3 + [0.0] * 3
        yaw = sequence[min(count, len(sequence) - 1)]
        self._diagnostic_pose_count = count + 1
        return True, yaw

    def process(self, rgb, identity, verified, overrides=None):
        start = time.perf_counter()
        out = rgb.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 42), (12, 10, 28), -1)
        cv2.putText(out, "DIAGNOSTIC STREAM - GPU MODEL NOT MOUNTED", (14, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.42, w / 1500), (205, 190, 255), 1, cv2.LINE_AA)
        return FrameResult(out, True, True, (time.perf_counter() - start) * 1000)
