"""Sub-symbolic common-sense gap filling via memory interpolation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CommonSenseReasoner:
    """Fills missing latent dimensions using nearest-memory interpolation."""

    memory_size: int = 512
    noise_floor: float = 1e-6
    memory: list[np.ndarray] = field(default_factory=list)

    def remember(self, latent: np.ndarray) -> None:
        latent = np.asarray(latent, dtype=np.float64).ravel()
        self.memory.append(latent)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size :]

    def fill_gaps(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float64).ravel()
        if not self.memory:
            return latent

        missing = ~np.isfinite(latent)
        if not np.any(missing):
            return latent

        observed = ~missing
        if not np.any(observed):
            return np.nan_to_num(np.mean(np.asarray(self.memory), axis=0), nan=0.0)

        memory_array = np.asarray(self.memory)
        reference = latent.copy()
        reference[missing] = 0.0

        distances = np.linalg.norm(memory_array[:, observed] - reference[observed], axis=1)
        weights = np.exp(-distances / (np.std(distances) + self.noise_floor))
        weights /= np.sum(weights) + self.noise_floor

        estimate = np.sum(memory_array * weights[:, None], axis=0)
        filled = latent.copy()
        filled[missing] = estimate[missing]
        return filled
