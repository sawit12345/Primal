"""Bilateral hemifield split and pull imbalance module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BilateralHemifield:
    """Splits latent coordinates into bilateral hemifields with pull imbalance."""

    imbalance: float = 0.08

    def split(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(features, dtype=np.float64).ravel()
        midpoint = vector.size // 2
        left = vector[:midpoint]
        right = vector[midpoint:]
        return left, right

    def integrate(self, features: np.ndarray) -> np.ndarray:
        left, right = self.split(features)
        if left.size == 0 or right.size == 0:
            return np.asarray(features, dtype=np.float64).ravel()

        left_pull = left * (1.0 + self.imbalance)
        right_pull = right * (1.0 - self.imbalance)
        return np.concatenate([left_pull, right_pull], axis=0)
