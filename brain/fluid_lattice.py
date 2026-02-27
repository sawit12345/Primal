"""Lightweight 18-step lattice-Boltzmann intuitive physics module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatticeBoltzmannIntuition:
    """D2Q9 LBM used as a latent intuitive-physics prior generator."""

    height: int = 16
    width: int = 16
    tau: float = 0.8
    f: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.c = np.array(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
            dtype=np.int64,
        )
        self.w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
        self.f = np.zeros((9, self.height, self.width), dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        rho = np.ones((self.height, self.width), dtype=np.float64)
        ux = np.zeros_like(rho)
        uy = np.zeros_like(rho)
        self.f = self._equilibrium(rho, ux, uy)

    def _equilibrium(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        cu = np.zeros((9, self.height, self.width), dtype=np.float64)
        u_sq = ux**2 + uy**2
        for i, (cx, cy) in enumerate(self.c):
            cu[i] = 3.0 * (cx * ux + cy * uy)
        feq = np.zeros_like(cu)
        for i in range(9):
            feq[i] = self.w[i] * rho * (1.0 + cu[i] + 0.5 * cu[i] ** 2 - 1.5 * u_sq)
        return feq

    def inject_velocity(self, vx: float, vy: float) -> None:
        rho = np.sum(self.f, axis=0)
        ux = np.full((self.height, self.width), vx, dtype=np.float64)
        uy = np.full((self.height, self.width), vy, dtype=np.float64)
        self.f = self._equilibrium(rho, ux, uy)

    def _collision_and_stream(self) -> None:
        rho = np.sum(self.f, axis=0)
        rho = np.maximum(rho, 1e-8)
        ux = np.sum(self.f * self.c[:, 0, None, None], axis=0) / rho
        uy = np.sum(self.f * self.c[:, 1, None, None], axis=0) / rho

        feq = self._equilibrium(rho, ux, uy)
        self.f += -(self.f - feq) / self.tau

        streamed = np.zeros_like(self.f)
        for i, (cx, cy) in enumerate(self.c):
            streamed[i] = np.roll(np.roll(self.f[i], shift=cx, axis=1), shift=cy, axis=0)
        self.f = streamed

    def advance_18_steps(self) -> np.ndarray:
        for _ in range(18):
            self._collision_and_stream()

        rho = np.sum(self.f, axis=0)
        rho = np.maximum(rho, 1e-8)
        ux = np.sum(self.f * self.c[:, 0, None, None], axis=0) / rho
        uy = np.sum(self.f * self.c[:, 1, None, None], axis=0) / rho
        velocity = np.stack([ux.mean(), uy.mean()], axis=0)
        return velocity
