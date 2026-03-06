"""Continuous Gaussian proprioception model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.temporal_decay import MarkovTemporalDecay


@dataclass
class ProprioceptiveGaussian:
    """Online Gaussian body-state estimator with uncertainty."""

    dim: int
    decay: MarkovTemporalDecay = field(default_factory=MarkovTemporalDecay)
    ridge: float = 1e-3
    mean: np.ndarray = field(init=False)
    covariance: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.covariance = np.eye(self.dim, dtype=np.float64)

    def update(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape[0] != self.dim:
            raise ValueError(f"Expected proprioceptive state dim={self.dim}, got {state.shape[0]}.")

        delta = state - self.mean
        sample_cov = np.outer(delta, delta)

        self.mean = self.decay.blend(self.mean, state)
        blended_cov = self.decay.blend(self.covariance, sample_cov)
        self.covariance = blended_cov + self.ridge * np.eye(self.dim, dtype=np.float64)

    def surprise(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=np.float64)
        delta = state - self.mean
        inv_cov = np.linalg.pinv(self.covariance)
        mahalanobis = float(delta.T @ inv_cov @ delta)
        sign, log_det = np.linalg.slogdet(self.covariance)
        if sign <= 0.0:
            log_det = 0.0
        return 0.5 * (mahalanobis + log_det + self.dim * np.log(2.0 * np.pi))
