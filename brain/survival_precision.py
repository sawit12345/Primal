"""Precision alpha scaling for survival urgency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurvivalUrgencyController:
    """Maps homeostatic stress into precision multipliers."""

    baseline_alpha: float = 1.0
    max_alpha: float = 8.0
    stress_gain: float = 4.0

    def alpha(self, homeostatic_error: float) -> float:
        stress = max(0.0, float(homeostatic_error))
        scaled = self.baseline_alpha + self.stress_gain * np.tanh(stress)
        return float(np.clip(scaled, self.baseline_alpha, self.max_alpha))
