"""Superior colliculus-inspired orienting and saliency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SuperiorColliculus:
    """Computes saliency-driven orienting vectors from latent activity."""

    gain: float = 1.0

    def saliency(self, features: np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=np.float64).ravel()
        centered = vector - np.mean(vector)
        return np.abs(centered)

    def orient(self, features: np.ndarray) -> np.ndarray:
        saliency = self.saliency(features)
        if saliency.size == 0:
            return np.zeros(2, dtype=np.float64)

        idx = int(np.argmax(saliency))
        normalized_idx = idx / max(1, saliency.size - 1)
        horizontal = 2.0 * normalized_idx - 1.0
        magnitude = float(np.max(saliency))
        return self.gain * np.array([horizontal, magnitude], dtype=np.float64)
