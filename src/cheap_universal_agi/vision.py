from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage, signal

from .config import VisionConfig


def _as_float_image(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float32)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _resize(frame: np.ndarray, side: int) -> np.ndarray:
    if frame.shape[0] == side and frame.shape[1] == side:
        return frame
    zoom_h = side / frame.shape[0]
    zoom_w = side / frame.shape[1]
    return ndimage.zoom(frame, (zoom_h, zoom_w, 1.0), order=1)


def _gabor_kernel(
    theta: float,
    frequency: float,
    sigma_x: float,
    sigma_y: float,
    size: int,
    phase: float,
) -> np.ndarray:
    center = size // 2
    yy, xx = np.mgrid[-center : center + 1, -center : center + 1]
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    envelope = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    harmonic = np.cos(2.0 * np.pi * frequency * xr + phase)
    kernel = envelope * harmonic
    kernel -= kernel.mean()
    norm = np.linalg.norm(kernel) + 1e-8
    return kernel / norm


@dataclass(slots=True)
class RetinaOutput:
    channels: np.ndarray  # [4, H, W]
    edge_density: np.ndarray  # [H, W]


class RetinaV1Pipeline:
    """
    Retinal opponency + DoG + fixed V1 Gabor bank + V5 motion.

    This is intentionally parameter-light and follows BLUEPRINT.md constraints:
    - Color opponency channels before DoG.
    - DoG (sigma 1 and 3).
    - 8 orientations, 3 spatial scales, 2 phases.
    - Hard sparsity threshold at 0.2 * max per filter map.
    """

    def __init__(self, cfg: VisionConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._kernels = self._build_gabor_bank()
        self._prev_v1_energy: np.ndarray | None = None

    def _build_gabor_bank(self) -> list[np.ndarray]:
        kernels: list[np.ndarray] = []
        orientations = np.linspace(
            0.0, np.pi, self.cfg.gabor_orientations, endpoint=False, dtype=np.float32
        )
        # Convert "cycles per degree" into normalized image-frequency proxies.
        # Using 84x84 or 128x128 retinal grids, these ratios preserve low/mid/high bands.
        freq_norm = [f / (self.cfg.input_size / 2.0) for f in self.cfg.gabor_scales]
        for theta in orientations:
            for f in freq_norm:
                sigma = max(1.2, 1.0 / max(f, 1e-3))
                kernels.append(
                    _gabor_kernel(
                        theta=theta,
                        frequency=f,
                        sigma_x=sigma,
                        sigma_y=sigma * 0.8,
                        size=self.cfg.gabor_kernel_size,
                        phase=0.0,
                    )
                )
                kernels.append(
                    _gabor_kernel(
                        theta=theta,
                        frequency=f,
                        sigma_x=sigma,
                        sigma_y=sigma * 0.8,
                        size=self.cfg.gabor_kernel_size,
                        phase=np.pi / 2.0,
                    )
                )
        return kernels

    def retina(self, frame: np.ndarray) -> RetinaOutput:
        x = _as_float_image(frame)
        x = _resize(x, self.cfg.input_size)
        r = x[..., 0]
        g = x[..., 1]
        b = x[..., 2]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        yellow = 0.5 * (r + g)
        rg = r - g
        by = b - yellow

        y_narrow = ndimage.gaussian_filter(y, sigma=self.cfg.dog_sigma_narrow)
        y_wide = ndimage.gaussian_filter(y, sigma=self.cfg.dog_sigma_wide)
        dog = y_narrow - y_wide
        on = np.clip(dog, 0.0, None)
        off = np.clip(-dog, 0.0, None)

        channels = np.stack([on, off, rg, by], axis=0).astype(np.float32)
        edge_density = np.abs(dog).astype(np.float32)
        return RetinaOutput(channels=channels, edge_density=edge_density)

    def v1(self, retinal: RetinaOutput) -> np.ndarray:
        channel_maps: list[np.ndarray] = []
        for kernel in self._kernels:
            # Apply each filter to each of the 4 retinal channels and sum magnitudes.
            responses = []
            for c in range(retinal.channels.shape[0]):
                resp = signal.fftconvolve(retinal.channels[c], kernel, mode="same")
                responses.append(resp)
            stacked = np.stack(responses, axis=0)
            energy = np.sqrt(np.sum(stacked**2, axis=0))
            threshold = self.cfg.v1_threshold_frac * (np.max(energy) + 1e-8)
            sparse = np.where(energy >= threshold, energy, 0.0)
            channel_maps.append(sparse.astype(np.float32))
        return np.stack(channel_maps, axis=0)

    def v5_motion(self, current_v1: np.ndarray) -> np.ndarray:
        """
        Motion field approximation using inter-frame V1 energy displacement.
        Returns [grid, grid, 2] field (vx, vy).
        """
        energy = current_v1.mean(axis=0)
        if self._prev_v1_energy is None:
            self._prev_v1_energy = energy.copy()
            return np.zeros((self.cfg.v5_grid, self.cfg.v5_grid, 2), dtype=np.float32)

        diff = energy - self._prev_v1_energy
        gx = ndimage.sobel(diff, axis=1)
        gy = ndimage.sobel(diff, axis=0)
        self._prev_v1_energy = energy.copy()

        block_h = energy.shape[0] // self.cfg.v5_grid
        block_w = energy.shape[1] // self.cfg.v5_grid
        flow = np.zeros((self.cfg.v5_grid, self.cfg.v5_grid, 2), dtype=np.float32)
        for i in range(self.cfg.v5_grid):
            for j in range(self.cfg.v5_grid):
                ys = slice(i * block_h, (i + 1) * block_h)
                xs = slice(j * block_w, (j + 1) * block_w)
                flow[i, j, 0] = gx[ys, xs].mean()
                flow[i, j, 1] = gy[ys, xs].mean()
        return flow

    def superior_colliculus_salience(
        self,
        edge_density: np.ndarray,
        flow: np.ndarray,
        threat_scalar: float = 0.0,
        edge_weight: float = 0.3,
        motion_weight: float = 0.3,
        threat_weight: float = 0.4,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        edge_grid = self._pool_to_grid(edge_density, self.cfg.v5_grid)
        motion_mag = np.linalg.norm(flow, axis=-1)
        edge_norm = edge_grid / (edge_grid.max() + 1e-8)
        motion_norm = motion_mag / (motion_mag.max() + 1e-8)
        threat_map = np.full_like(edge_norm, np.clip(threat_scalar, 0.0, 1.0))
        salience = (
            edge_weight * edge_norm + motion_weight * motion_norm + threat_weight * threat_map
        )
        flat_idx = int(np.argmax(salience))
        i, j = np.unravel_index(flat_idx, salience.shape)
        return salience.astype(np.float32), (int(i), int(j))

    @staticmethod
    def _pool_to_grid(arr: np.ndarray, grid: int) -> np.ndarray:
        h, w = arr.shape
        bh, bw = h // grid, w // grid
        out = np.zeros((grid, grid), dtype=np.float32)
        for i in range(grid):
            for j in range(grid):
                out[i, j] = arr[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw].mean()
        return out

    @staticmethod
    def flatten_v1(v1_maps: np.ndarray) -> np.ndarray:
        return v1_maps.reshape(-1).astype(np.float32)

    @staticmethod
    def sparse_topk(x: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros_like(x)
        if k <= 0:
            return out
        idx = np.argpartition(np.abs(x), -k)[-k:]
        out[idx] = x[idx]
        return out
