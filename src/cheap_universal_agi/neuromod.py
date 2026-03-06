from __future__ import annotations

import numpy as np


class Neuromodulators:
    def __init__(self):
        self.avg_reward_interval = 20.0
        self.steps_since_reward = 0.0
        self.last_gamma = 0.95

    def dopamine(self, td_error: float) -> float:
        if td_error > 0.0:
            self.steps_since_reward = 0.0
        else:
            self.steps_since_reward += 1.0
        self.avg_reward_interval = 0.99 * self.avg_reward_interval + 0.01 * self.steps_since_reward
        return float(td_error)

    def norepinephrine(self, cortical_errors: np.ndarray) -> float:
        return float(np.clip(np.mean(np.abs(cortical_errors)), 0.0, 1.0))

    def serotonin_gamma(self) -> float:
        # Frequent rewards -> lower gamma; sparse rewards -> higher gamma.
        x = np.clip(self.avg_reward_interval / 50.0, 0.0, 1.0)
        gamma = 0.9 + 0.1 * x
        self.last_gamma = gamma
        return float(gamma)

    def acetylcholine(self, retrieval_scores: np.ndarray) -> float:
        if retrieval_scores.size == 0:
            return 0.0
        p = retrieval_scores.astype(np.float64)
        p = p / (p.sum() + 1e-9)
        entropy = -(p * np.log(p + 1e-12)).sum()
        max_entropy = np.log(len(p) + 1e-12)
        return float(np.clip(entropy / (max_entropy + 1e-8), 0.0, 1.0))
