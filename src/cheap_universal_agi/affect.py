from __future__ import annotations

import numpy as np


class AmygdalaValence:
    """
    512 -> 2 linear readout:
    - threat scalar
    - reward anticipation scalar
    """

    def __init__(self, input_dim: int = 512, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(2, input_dim)).astype(np.float32)
        self.b = np.zeros(2, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[float, float]:
        y = self.W @ x.astype(np.float32) + self.b
        threat = float(1.0 / (1.0 + np.exp(-y[0])))
        reward = float(1.0 / (1.0 + np.exp(-y[1])))
        return threat, reward

    def update(self, x: np.ndarray, dopamine: float, lr: float = 0.01):
        target = np.array([1.0 if dopamine < 0 else 0.0, 1.0 if dopamine > 0 else 0.0], dtype=np.float32)
        y = self.W @ x + self.b
        pred = 1.0 / (1.0 + np.exp(-y))
        err = target - pred
        self.W += lr * np.outer(err, x).astype(np.float32)
        self.b += lr * err.astype(np.float32)
