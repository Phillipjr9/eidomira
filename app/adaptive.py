from __future__ import annotations

from collections import deque
import numpy as np


PRESETS = {
    "speed": {"max_width": 512, "target_ms": 32.0, "temporal": .14},
    "balanced": {"max_width": 768, "target_ms": 45.0, "temporal": .22},
    "quality": {"max_width": 960, "target_ms": 65.0, "temporal": .27},
}


class AdaptiveQualityController:
    """Latency controller that changes inference resolution, never queue depth."""

    def __init__(self, preset: str, hard_max: int, minimum: int, interval: int):
        config = PRESETS.get(preset, PRESETS["balanced"])
        self.preset = preset if preset in PRESETS else "balanced"
        self.maximum = min(hard_max, config["max_width"])
        self.minimum = min(minimum, self.maximum)
        self.width = self.maximum
        self.target = config["target_ms"]
        self.temporal = config["temporal"]
        self.interval = max(8, interval)
        self.samples = deque(maxlen=self.interval)
        self.adjustments = 0

    def observe(self, latency_ms: float) -> int:
        if latency_ms <= 0:
            return self.width
        self.samples.append(latency_ms)
        if len(self.samples) < self.interval:
            return self.width
        p80 = float(np.percentile(self.samples, 80))
        old = self.width
        if p80 > self.target * 1.25:
            self.width = max(self.minimum, int(self.width * .84) // 16 * 16)
        elif p80 < self.target * .66:
            self.width = min(self.maximum, int(self.width * 1.12) // 16 * 16)
        if self.width != old:
            self.adjustments += 1
            self.samples.clear()
        return self.width
