"""Renormalization-group-inspired multi-scale feature pooling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RenormalizationGroup:
    """Produces coarse-grained representations at multiple scales."""

    max_levels: int = 4

    def transform(self, features: np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=np.float64).ravel()
        levels: list[np.ndarray] = [vector]

        current = vector
        for _ in range(self.max_levels - 1):
            if current.size < 2:
                break
            if current.size % 2 != 0:
                current = np.pad(current, (0, 1), mode="edge")
            current = 0.5 * (current[0::2] + current[1::2])
            levels.append(current)

        normalized_levels = [lvl / (np.linalg.norm(lvl) + 1e-8) for lvl in levels]
        return np.concatenate(normalized_levels, axis=0)
