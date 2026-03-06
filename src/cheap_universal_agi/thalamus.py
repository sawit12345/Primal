from __future__ import annotations

import numpy as np


class ThalamicRouter:
    """
    Learned binary routing mask controlled by novelty and goals.
    """

    def __init__(self, input_dim: int = 640, hidden_dim: int = 128, seed: int = 0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(1 + input_dim + 64), size=(hidden_dim, 1 + input_dim + 64)).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(input_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(input_dim, dtype=np.float32)

    def mask(self, norepinephrine: float, cortical_error: np.ndarray, goal_vector: np.ndarray) -> np.ndarray:
        x = np.concatenate(
            [
                np.array([norepinephrine], dtype=np.float32),
                cortical_error.astype(np.float32),
                goal_vector.astype(np.float32),
            ]
        )
        h = np.maximum(0.0, self.W1 @ x + self.b1)
        logits = self.W2 @ h + self.b2
        # High norepinephrine opens more gates.
        threshold = 0.0 - 0.75 * float(np.clip(norepinephrine, 0.0, 1.0))
        return (logits > threshold).astype(np.float32)
