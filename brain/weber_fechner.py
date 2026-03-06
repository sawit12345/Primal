"""Weber-Fechner logarithmic precision scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeberFechnerANS:
    """Adaptive number sense precision scaling via logarithmic compression."""

    base_precision: float = 1.0
    just_noticeable_difference: float = 1e-3
    gain: float = 1.0

    def scale(self, magnitude: np.ndarray | float) -> np.ndarray:
        magnitude_array = np.asarray(magnitude, dtype=np.float64)
        compressed = np.log1p(np.abs(magnitude_array) / self.just_noticeable_difference)
        return self.base_precision * (1.0 + self.gain * compressed)
