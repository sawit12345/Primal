"""Markovian temporal decay utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarkovTemporalDecay:
    """Applies first-order temporal blending to running beliefs.

    The default update is exactly the requested 0.7 old + 0.3 new.
    """

    old_weight: float = 0.7
    new_weight: float = 0.3

    def __post_init__(self) -> None:
        total = self.old_weight + self.new_weight
        if total <= 0.0:
            raise ValueError("Temporal decay weights must sum to a positive value.")
        self.old_weight /= total
        self.new_weight /= total

    def blend(self, old_value: np.ndarray, new_value: np.ndarray) -> np.ndarray:
        return self.old_weight * old_value + self.new_weight * new_value

    def blend_scalar(self, old_value: float, new_value: float) -> float:
        return float(self.old_weight * old_value + self.new_weight * new_value)
