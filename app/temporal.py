from __future__ import annotations

import cv2
import numpy as np


class MotionAwareStabilizer:
    """Suppresses frame-to-frame neural shimmer without smearing fast motion.

    It compares consecutive input frames, builds a soft motion confidence map,
    and only reuses previous output pixels where the camera image is stable.
    """

    def __init__(self, strength: float = .22, motion_threshold: float = 24.0):
        self.strength = float(np.clip(strength, 0, .65))
        self.threshold = max(1.0, float(motion_threshold))
        self.previous_input: np.ndarray | None = None
        self.previous_output: np.ndarray | None = None

    def reset(self):
        self.previous_input = self.previous_output = None

    def apply(self, input_rgb: np.ndarray, output_rgb: np.ndarray) -> np.ndarray:
        if (self.previous_input is None or self.previous_input.shape != input_rgb.shape
                or self.previous_output is None or self.previous_output.shape != output_rgb.shape):
            self.previous_input = input_rgb.copy()
            self.previous_output = output_rgb.copy()
            return output_rgb

        current_gray = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY)
        previous_gray = cv2.cvtColor(self.previous_input, cv2.COLOR_RGB2GRAY)
        motion = cv2.absdiff(current_gray, previous_gray).astype(np.float32)
        motion = cv2.GaussianBlur(motion, (9, 9), 0)
        # 1 in static areas, smoothly approaching 0 around movement.
        stable = np.clip(1.0 - motion / self.threshold, 0.0, 1.0)
        weight = (stable * self.strength)[..., None]
        blended = output_rgb.astype(np.float32) * (1.0 - weight)
        blended += self.previous_output.astype(np.float32) * weight
        result = np.clip(blended, 0, 255).astype(np.uint8)
        self.previous_input = input_rgb.copy()
        self.previous_output = result.copy()
        return result
