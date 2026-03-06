"""Occipital-lobe-inspired visual primitive extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OccipitalLobe:
    """Extracts edge and motion-like primitives from vector or image inputs."""

    downsample: int = 2

    def encode(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.ndim == 1:
            gradient = np.diff(obs, prepend=obs[0])
            laplacian = np.diff(gradient, prepend=gradient[0])
            return np.concatenate([obs, gradient, laplacian], axis=0)

        if obs.ndim == 2:
            gx = np.diff(obs, axis=1, prepend=obs[:, :1])
            gy = np.diff(obs, axis=0, prepend=obs[:1, :])
            magnitude = np.sqrt(gx**2 + gy**2)
            pooled = magnitude[:: self.downsample, :: self.downsample]
            return np.concatenate([pooled.ravel(), gx.ravel(), gy.ravel()], axis=0)

        flat = obs.ravel()
        return self.encode(flat)
