"""Cerebellar-inspired action smoothing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CerebellarSmoother:
    """Smooths control trajectories while preserving responsiveness."""

    smoothing: float = 0.75
    damping: float = 0.15
    previous_action: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    previous_velocity: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))

    def reset(self, action_dim: int) -> None:
        self.previous_action = np.zeros(action_dim, dtype=np.float64)
        self.previous_velocity = np.zeros(action_dim, dtype=np.float64)

    def smooth(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.previous_action.shape:
            self.reset(int(action.shape[0]))

        velocity = action - self.previous_action
        damped_velocity = (1.0 - self.damping) * velocity + self.damping * self.previous_velocity
        smoothed = self.smoothing * self.previous_action + (1.0 - self.smoothing) * (
            self.previous_action + damped_velocity
        )

        self.previous_velocity = damped_velocity
        self.previous_action = smoothed
        return smoothed
