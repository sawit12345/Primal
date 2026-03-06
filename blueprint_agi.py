#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import ale_py
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal
from skimage.feature import hog
from skimage.transform import AffineTransform, resize, warp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_openml
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

gym.register_envs(ale_py)


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]


def softmax(logits: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    probs = np.exp(logits)
    probs_sum = np.sum(probs, axis=axis, keepdims=True) + 1e-8
    return probs / probs_sum


def l2_normalize(x: NDArray[np.float32], axis: int = -1) -> NDArray[np.float32]:
    denom = np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8
    return (x / denom).astype(np.float32)


def cosine_similarity(a: FloatArray, b: FloatArray) -> FloatArray:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return (a_n @ b_n.T).astype(np.float32)


def one_hot(index: int, size: int) -> FloatArray:
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def topk_binary(x: FloatArray, frac: float) -> FloatArray:
    keep = max(1, int(round(x.size * frac)))
    idx = np.argpartition(x, -keep)[-keep:]
    out = np.zeros_like(x, dtype=np.float32)
    out[idx] = 1.0
    return out


def reflect_position(x: float, low: float, high: float) -> float:
    span = high - low
    if span <= 0:
        return low
    while x < low or x > high:
        if x < low:
            x = low + (low - x)
        if x > high:
            x = high - (x - high)
    return x


@dataclass
class VisionConfig:
    retina_sigma_narrow: float = 1.0
    retina_sigma_wide: float = 3.0
    input_size: int = 32
    v1_threshold_ratio: float = 0.2
    gabor_orientations: int = 8
    gabor_frequencies: tuple[float, float, float] = (0.08, 0.16, 0.32)
    pooled_grid: int = 16


class RetinaEarlyVision:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def preprocess(self, rgb: FloatArray) -> FloatArray:
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        if rgb.shape[0] != self.config.input_size or rgb.shape[1] != self.config.input_size:
            rgb = resize(
                rgb,
                (self.config.input_size, self.config.input_size),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)

        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        rg = r - g
        by = b - 0.5 * (r + g)

        narrow = ndimage.gaussian_filter(luminance, sigma=self.config.retina_sigma_narrow)
        wide = ndimage.gaussian_filter(luminance, sigma=self.config.retina_sigma_wide)
        dog = narrow - wide
        on = np.clip(dog, 0.0, None)
        off = np.clip(-dog, 0.0, None)

        rg = rg - rg.mean()
        by = by - by.mean()
        rg /= rg.std() + 1e-6
        by /= by.std() + 1e-6

        return np.stack([on, off, rg, by], axis=0).astype(np.float32)


class V1GaborBank:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.orientations = np.linspace(0.0, np.pi, config.gabor_orientations, endpoint=False).astype(np.float32)
        # These sigmas approximate the coarse-to-fine spatial frequency stack while avoiding
        # a full FFT convolution bank for every image.
        self.scale_sigmas = np.array([0.8, 1.6, 3.2], dtype=np.float32)

    def forward(self, retina_maps: FloatArray) -> FloatArray:
        responses = []
        for channel_index in range(retina_maps.shape[0]):
            channel = retina_maps[channel_index]
            for sigma in self.scale_sigmas.tolist():
                blurred = ndimage.gaussian_filter(channel, sigma=sigma).astype(np.float32)
                gx = ndimage.sobel(blurred, axis=1).astype(np.float32)
                gy = ndimage.sobel(blurred, axis=0).astype(np.float32)
                for theta in self.orientations.tolist():
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)
                    phase0 = cos_t * gx + sin_t * gy
                    phase90 = -sin_t * gx + cos_t * gy
                    for conv in (phase0, phase90):
                        max_abs = np.max(np.abs(conv)) + 1e-8
                        sparse = conv.copy()
                        sparse[np.abs(sparse) < self.config.v1_threshold_ratio * max_abs] = 0.0
                        responses.append(sparse.astype(np.float32))
        return np.stack(responses, axis=0).astype(np.float32)

    def pooled_state_grid(self, v1_maps: FloatArray) -> IntArray:
        c, h, w = v1_maps.shape
        grid = self.config.pooled_grid
        pooled = v1_maps.reshape(c, grid, h // grid, grid, w // grid).mean(axis=(2, 4))
        state_grid = np.argmax(pooled, axis=0).astype(np.int32)
        return state_grid

    def edge_density(self, v1_maps: FloatArray) -> FloatArray:
        energy = np.mean(np.abs(v1_maps), axis=0)
        grid = 16
        pooled = energy.reshape(grid, energy.shape[0] // grid, grid, energy.shape[1] // grid).mean(axis=(1, 3))
        pooled = pooled - pooled.min()
        pooled /= pooled.max() + 1e-8
        return pooled.astype(np.float32)


@dataclass
class RGMLevelSpec:
    max_states: int
    child_vocab_size: int
    child_positions: int
    path_count: int = 4
    pseudocount: float = 1e-2
    novelty_threshold: float = -14.0


class DiscreteRGMLevel:
    def __init__(self, spec: RGMLevelSpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self.rng = rng
        self.state_count = 0
        self.counts = np.full(
            (spec.max_states, spec.child_positions, spec.child_vocab_size),
            spec.pseudocount,
            dtype=np.float32,
        )
        self.prior_counts = np.full(spec.max_states, spec.pseudocount, dtype=np.float32)
        self.transition_counts = np.full(
            (spec.max_states, spec.max_states, spec.path_count),
            spec.pseudocount,
            dtype=np.float32,
        )

    @property
    def active_counts(self) -> FloatArray:
        return self.counts[: self.state_count]

    def _log_state_likelihood(self, groups: IntArray) -> NDArray[np.float64]:
        active_counts = self.counts[: self.state_count]
        if self.state_count == 0:
            return np.empty((len(groups), 0), dtype=np.float64)
        probs = active_counts / (active_counts.sum(axis=2, keepdims=True) + 1e-8)
        log_probs = np.log(probs + 1e-8)
        ll = np.zeros((len(groups), self.state_count), dtype=np.float64)
        for pos in range(self.spec.child_positions):
            ll += log_probs[:, pos, groups[:, pos]].T
        priors = np.log(
            self.prior_counts[: self.state_count] / (np.sum(self.prior_counts[: self.state_count]) + 1e-8) + 1e-8
        )
        ll += priors[None, :]
        return ll

    def _create_state(self, group: IntArray, weight: float) -> int:
        idx = self.state_count
        self.state_count += 1
        self.prior_counts[idx] += weight
        for pos, child_state in enumerate(group.tolist()):
            self.counts[idx, pos, child_state] += weight * 8.0
        return idx

    def _update_state(self, state: int, group: IntArray, weight: float) -> None:
        self.prior_counts[state] += weight
        for pos, child_state in enumerate(group.tolist()):
            self.counts[state, pos, child_state] += weight

    def fit_groups(self, groups: IntArray, sample_weights: FloatArray | None = None) -> IntArray:
        if sample_weights is None:
            sample_weights = np.ones(len(groups), dtype=np.float32)

        unique_groups, inverse, counts = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
        weighted_counts = np.zeros(len(unique_groups), dtype=np.float32)
        for idx, inv in enumerate(inverse):
            weighted_counts[inv] += sample_weights[idx]

        order = np.argsort(-weighted_counts)
        group_to_state: dict[tuple[int, ...], int] = {}
        for row_idx in order.tolist():
            group = unique_groups[row_idx]
            weight = float(weighted_counts[row_idx])
            if self.state_count == 0:
                state = self._create_state(group, weight)
                group_to_state[tuple(group.tolist())] = state
                continue

            ll = self._log_state_likelihood(group[None, :])[0]
            best_state = int(np.argmax(ll))
            best_score = float(ll[best_state])
            if best_score < self.spec.novelty_threshold and self.state_count < self.spec.max_states:
                state = self._create_state(group, weight)
            else:
                state = best_state
                self._update_state(state, group, weight)
            group_to_state[tuple(group.tolist())] = state

        assigned = np.array([group_to_state[tuple(row.tolist())] for row in groups], dtype=np.int32)
        return assigned

    def infer_posteriors(self, groups: IntArray) -> FloatArray:
        ll = self._log_state_likelihood(groups)
        if ll.shape[1] == 0:
            return np.zeros((len(groups), self.spec.max_states), dtype=np.float32)
        probs = softmax(ll, axis=1).astype(np.float32)
        full = np.zeros((len(groups), self.spec.max_states), dtype=np.float32)
        full[:, : self.state_count] = probs
        return full

    def infer_states(self, groups: IntArray) -> tuple[IntArray, FloatArray]:
        post = self.infer_posteriors(groups)
        return np.argmax(post, axis=1).astype(np.int32), post

    def update_transitions(self, states_prev: IntArray, states_next: IntArray, paths: IntArray) -> None:
        if len(states_prev) == 0:
            return
        for s0, s1, path in zip(states_prev.tolist(), states_next.tolist(), paths.tolist()):
            self.transition_counts[s0, s1, int(path)] += 1.0


def group_grid_states(state_grid: IntArray, group_shape: tuple[int, int]) -> IntArray:
    gh, gw = group_shape
    h, w = state_grid.shape
    out_h, out_w = h // gh, w // gw
    reshaped = state_grid.reshape(out_h, gh, out_w, gw).transpose(0, 2, 1, 3)
    return reshaped.reshape(out_h * out_w, gh * gw).astype(np.int32)


class VisualRGMHierarchy:
    def __init__(self, rng: np.random.Generator, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self.retina = RetinaEarlyVision(self.config)
        self.v1 = V1GaborBank(self.config)
        self.level1 = DiscreteRGMLevel(
            RGMLevelSpec(max_states=64, child_vocab_size=96, child_positions=4, path_count=4, novelty_threshold=-18.0),
            rng,
        )
        self.level2 = DiscreteRGMLevel(
            RGMLevelSpec(max_states=128, child_vocab_size=64, child_positions=4, path_count=4, novelty_threshold=-12.0),
            rng,
        )
        self.level3 = DiscreteRGMLevel(
            RGMLevelSpec(max_states=256, child_vocab_size=128, child_positions=16, path_count=4, novelty_threshold=-48.0),
            rng,
        )

    def frontend(self, image: FloatArray) -> tuple[FloatArray, FloatArray, IntArray]:
        retina_maps = self.retina.preprocess(image)
        v1_maps = self.v1.forward(retina_maps)
        grid = self.v1.pooled_state_grid(v1_maps)
        return retina_maps, v1_maps, grid

    def fit(self, images: FloatArray) -> None:
        level1_groups: list[IntArray] = []
        level2_grids: list[IntArray] = []
        level2_groups: list[IntArray] = []
        level3_groups: list[IntArray] = []

        for image in images:
            _, _, obs_grid = self.frontend(image)
            l1_groups = group_grid_states(obs_grid, (2, 2))
            level1_groups.append(l1_groups)

        all_l1_groups = np.concatenate(level1_groups, axis=0)
        all_l1_states = self.level1.fit_groups(all_l1_groups)

        cursor = 0
        for l1_groups in level1_groups:
            count = len(l1_groups)
            l1_states = all_l1_states[cursor : cursor + count]
            cursor += count
            l1_grid = l1_states.reshape(8, 8)
            l2_groups = group_grid_states(l1_grid, (2, 2))
            level2_groups.append(l2_groups)
            level2_grids.append(l1_grid)

        all_l2_groups = np.concatenate(level2_groups, axis=0)
        all_l2_states = self.level2.fit_groups(all_l2_groups)

        cursor = 0
        for l2_groups in level2_groups:
            count = len(l2_groups)
            l2_states = all_l2_states[cursor : cursor + count]
            cursor += count
            l2_grid = l2_states.reshape(4, 4)
            l3_group = l2_grid.reshape(1, 16)
            level3_groups.append(l3_group)

        all_l3_groups = np.concatenate(level3_groups, axis=0)
        self.level3.fit_groups(all_l3_groups)

    def encode(self, image: FloatArray) -> dict[str, FloatArray | IntArray]:
        retina_maps, v1_maps, obs_grid = self.frontend(image)
        l1_groups = group_grid_states(obs_grid, (2, 2))
        l1_states, l1_post = self.level1.infer_states(l1_groups)
        l1_grid = l1_states.reshape(8, 8)

        l2_groups = group_grid_states(l1_grid, (2, 2))
        l2_states, l2_post = self.level2.infer_states(l2_groups)
        l2_grid = l2_states.reshape(4, 4)

        l3_group = l2_grid.reshape(1, 16)
        l3_states, l3_post = self.level3.infer_states(l3_group)
        top_state = int(l3_states[0])
        top_post = l3_post[0]

        l1_hist = np.bincount(l1_states, minlength=64).astype(np.float32)
        l2_hist = np.bincount(l2_states, minlength=128).astype(np.float32)
        visual_bus = np.concatenate([top_post, one_hot(top_state, 256)], dtype=np.float32)
        descriptor = np.concatenate([visual_bus, l1_hist, l2_hist], dtype=np.float32)

        return {
            "retina": retina_maps,
            "v1": v1_maps,
            "obs_grid": obs_grid,
            "l1_states": l1_states,
            "l2_states": l2_states,
            "top_state": np.array([top_state], dtype=np.int32),
            "top_post": top_post.astype(np.float32),
            "visual_bus": visual_bus.astype(np.float32),
            "descriptor": descriptor.astype(np.float32),
        }


class AuditoryRGMHierarchy:
    def __init__(self, rng: np.random.Generator, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.rng = rng
        self.mel_filters = self._build_mel_filters(512, 128, sample_rate)
        self.level1 = DiscreteRGMLevel(
            RGMLevelSpec(max_states=64, child_vocab_size=128, child_positions=4, novelty_threshold=-12.0),
            rng,
        )
        self.level2 = DiscreteRGMLevel(
            RGMLevelSpec(max_states=128, child_vocab_size=64, child_positions=4, novelty_threshold=-12.0),
            rng,
        )

    @staticmethod
    def _hz_to_mel(hz: FloatArray) -> FloatArray:
        return (2595.0 * np.log10(1.0 + hz / 700.0)).astype(np.float32)

    @staticmethod
    def _mel_to_hz(mel: FloatArray) -> FloatArray:
        return (700.0 * (10 ** (mel / 2595.0) - 1.0)).astype(np.float32)

    def _build_mel_filters(self, n_fft: int, n_mels: int, sr: int) -> FloatArray:
        mel_points = np.linspace(self._hz_to_mel(np.array([0.0], dtype=np.float32))[0], self._hz_to_mel(np.array([sr / 2], dtype=np.float32))[0], n_mels + 2)
        hz_points = self._mel_to_hz(mel_points.astype(np.float32))
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(np.int32)
        filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(1, n_mels + 1):
            left, center, right = bins[i - 1], bins[i], bins[i + 1]
            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1
            filters[i - 1, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
            filters[i - 1, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
        return filters

    def mel_spectrum(self, waveform: FloatArray) -> FloatArray:
        _, _, spec = signal.stft(
            waveform.astype(np.float32),
            fs=self.sample_rate,
            nperseg=int(0.025 * self.sample_rate),
            noverlap=int(0.015 * self.sample_rate),
            nfft=512,
            boundary=None,
        )
        mag = np.abs(spec).astype(np.float32)
        mel = self.mel_filters @ mag
        mel = np.log1p(mel)
        return mel.mean(axis=1).astype(np.float32)


@dataclass
class DentateGyrus:
    rng: np.random.Generator
    input_dim: int = 384
    output_dim: int = 3840
    sparsity: float = 0.02
    projection: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.projection = self.rng.normal(0.0, 1.0 / math.sqrt(self.input_dim), size=(self.output_dim, self.input_dim)).astype(np.float32)

    def separate(self, visual_belief: FloatArray, auditory_state: FloatArray) -> FloatArray:
        concat = np.concatenate([visual_belief, auditory_state], dtype=np.float32)
        projected = self.projection @ concat
        return topk_binary(projected.astype(np.float32), self.sparsity)


@dataclass
class LMUEntorhinalIndex:
    history_steps: int = 200
    feature_dim: int = 256
    output_dim: int = 128
    kernel: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        t = np.linspace(-1.0, 1.0, self.history_steps, dtype=np.float32)
        basis = []
        per_group = self.output_dim // 16
        for degree in range(per_group):
            coeffs = np.zeros(degree + 1, dtype=np.float32)
            coeffs[-1] = 1.0
            basis.append(np.polynomial.legendre.legval(t, coeffs).astype(np.float32))
        basis = np.stack(basis, axis=0)
        kernel = np.repeat(basis, 16, axis=0)
        self.kernel = kernel[: self.output_dim]

    def encode(self, history: list[FloatArray]) -> FloatArray:
        if len(history) == 0:
            return np.zeros(self.output_dim, dtype=np.float32)
        recent = history[-self.history_steps :]
        if len(recent) < self.history_steps:
            pad = [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(self.history_steps - len(recent))]
            recent = pad + recent
        stacked = np.stack(recent, axis=0).astype(np.float32)
        groups = stacked.reshape(self.history_steps, 16, self.feature_dim // 16).mean(axis=2).T
        coeffs = np.sum(groups[:, None, :] * self.kernel.reshape(16, -1, self.history_steps), axis=2)
        return coeffs.reshape(-1).astype(np.float32)[: self.output_dim]


@dataclass
class HippocampalTuple:
    index: FloatArray
    content: FloatArray
    goal: FloatArray
    context: FloatArray
    novelty: float
    social: bool = False
    retrievals: int = 0


class HippocampalBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.entries: list[HippocampalTuple] = []
        self.nn: NearestNeighbors | None = None
        self.index_matrix: FloatArray | None = None
        self.centroids: FloatArray | None = None

    def _rebuild_index(self) -> None:
        if not self.entries:
            self.nn = None
            self.index_matrix = None
            return
        self.index_matrix = np.stack([entry.index for entry in self.entries], axis=0).astype(np.float32)
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.index_matrix)

    def write(self, entry: HippocampalTuple) -> None:
        if len(self.entries) >= self.capacity:
            scores = np.array([0.5 * i / max(1, len(self.entries) - 1) + 0.5 * e.retrievals for i, e in enumerate(self.entries)], dtype=np.float32)
            drop = int(np.argmin(scores))
            self.entries.pop(drop)
        self.entries.append(entry)
        self._rebuild_index()

    def retrieve(self, query: FloatArray, top_k: int = 5) -> tuple[FloatArray, float]:
        if self.nn is None or not self.entries:
            return np.zeros(1152, dtype=np.float32), 0.0
        distances, indices = self.nn.kneighbors(query[None, :], n_neighbors=min(top_k, len(self.entries)))
        idx = int(indices[0, 0])
        confidence = float(1.0 - distances[0, 0])
        self.entries[idx].retrievals += 1
        return self.entries[idx].content.astype(np.float32), confidence

    def compress_level3(self, n_clusters: int = 16) -> None:
        if len(self.entries) < n_clusters:
            return
        matrix = np.stack([entry.content for entry in self.entries], axis=0).astype(np.float32)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = km.fit_predict(matrix[:, 256:1024])
        self.centroids = km.cluster_centers_.astype(np.float32)
        for entry, label in zip(self.entries, labels.tolist()):
            centroid = self.centroids[label]
            residual = entry.content.copy()
            residual[256:1024] -= centroid
            entry.content = residual.astype(np.float32)


class PredictiveCodingLayer:
    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator, learning_rate: float = 0.005) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.encoder = rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
        self.decoder = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(input_dim, hidden_dim)).astype(np.float32)
        self.hidden = np.zeros(hidden_dim, dtype=np.float32)

    def update(self, x: FloatArray, gate_threshold: float = 0.05) -> tuple[FloatArray, FloatArray, float]:
        prediction = self.decoder @ self.hidden
        error = x - prediction
        mean_abs_error = float(np.mean(np.abs(error)))
        if mean_abs_error > gate_threshold:
            hidden_delta = self.encoder @ error
            self.hidden = np.maximum(self.hidden + self.learning_rate * hidden_delta, 0.0).astype(np.float32)
            self.decoder += self.learning_rate * np.outer(error, self.hidden).astype(np.float32)
        return self.hidden.copy(), error.astype(np.float32), mean_abs_error


class RecurrentLayer6:
    def __init__(self, rng: np.random.Generator, input_dim: int = 128, hidden_dim: int = 256) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
        self.R = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.b = np.zeros(hidden_dim, dtype=np.float32)
        self.hidden = np.zeros(hidden_dim, dtype=np.float32)

    def step(self, x: FloatArray) -> FloatArray:
        self.hidden = np.maximum(self.W @ x + self.R @ self.hidden + self.b, 0.0).astype(np.float32)
        return self.hidden.copy()


class SuperiorColliculus:
    def __init__(self) -> None:
        self.weights = np.array([0.3, 0.3, 0.3], dtype=np.float32)

    def salience(self, edge_density: FloatArray, motion_map: FloatArray, threat_map: FloatArray | None = None) -> tuple[FloatArray, tuple[int, int]]:
        if threat_map is None:
            threat_map = np.zeros_like(edge_density, dtype=np.float32)
        stacked = np.stack([edge_density, motion_map, threat_map], axis=0)
        score = np.tensordot(self.weights, stacked, axes=(0, 0)).astype(np.float32)
        location = np.unravel_index(int(np.argmax(score)), score.shape)
        return score, (int(location[0]), int(location[1]))


class ParietalCortex:
    def __init__(self) -> None:
        self.grid = np.zeros((64, 64, 32), dtype=np.float32)

    def update(self, location: tuple[int, int], features: FloatArray) -> None:
        y = min(63, max(0, location[0] * 4))
        x = min(63, max(0, location[1] * 4))
        self.grid[y, x] = 0.9 * self.grid[y, x] + 0.1 * features[:32]


class Thalamus:
    def __init__(self, rng: np.random.Generator, input_dim: int = 640 + 64 + 1, hidden_dim: int = 128) -> None:
        self.W1 = rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
        self.W2 = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(5, hidden_dim)).astype(np.float32)

    def route(self, norepinephrine: float, cortical_error: FloatArray, goal: FloatArray) -> FloatArray:
        features = np.concatenate([np.array([norepinephrine], dtype=np.float32), cortical_error[:640], goal], dtype=np.float32)
        hidden = np.maximum(self.W1 @ features, 0.0)
        return (self.W2 @ hidden > 0.0).astype(np.float32)


class BasalGanglia:
    def __init__(self, action_dim: int = 64) -> None:
        self.action_dim = action_dim
        self.values = np.zeros(action_dim, dtype=np.float32)
        self.suppression = np.full(action_dim, 0.5, dtype=np.float32)
        self.eligibility = np.zeros(action_dim, dtype=np.float32)
        self.gamma = 0.95
        self.lambda_ = 0.9

    def select(self, action_scores: FloatArray) -> int:
        net = self.values + action_scores - self.suppression
        return int(np.argmax(net))

    def update(self, action: int, reward: float, next_value: float) -> float:
        td = reward + self.gamma * next_value - self.values[action]
        self.eligibility *= self.gamma * self.lambda_
        self.eligibility[action] += 1.0
        self.values += 0.05 * td * self.eligibility
        return float(td)


class Amygdala:
    def __init__(self, rng: np.random.Generator, input_dim: int = 512) -> None:
        self.W = rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(2, input_dim)).astype(np.float32)

    def readout(self, visual_belief: FloatArray, hippocampal_slice: FloatArray) -> tuple[float, float]:
        x = np.concatenate([visual_belief, hippocampal_slice], dtype=np.float32)
        valence = self.W @ x
        return float(valence[0]), float(valence[1])


class OrbitofrontalCortex:
    def __init__(self, rng: np.random.Generator) -> None:
        self.W1 = rng.normal(0.0, 1.0 / math.sqrt(259), size=(32, 259)).astype(np.float32)
        self.W2 = rng.normal(0.0, 1.0 / math.sqrt(32), size=(32, 32)).astype(np.float32)

    def evaluate(self, amygdala_pair: tuple[float, float], visual_belief: FloatArray, confidence: float) -> FloatArray:
        x = np.concatenate([np.array(amygdala_pair, dtype=np.float32), visual_belief, np.array([confidence], dtype=np.float32)])
        return np.maximum(self.W2 @ np.maximum(self.W1 @ x, 0.0), 0.0).astype(np.float32)


class Insula:
    def __init__(self) -> None:
        self.state = np.zeros(16, dtype=np.float32)

    def update(self, metrics: FloatArray) -> FloatArray:
        projection = np.linspace(-1.0, 1.0, 16 * len(metrics), dtype=np.float32).reshape(16, len(metrics))
        self.state = projection @ metrics.astype(np.float32)
        return self.state.copy()


class BlueprintCortex:
    def __init__(self, rng: np.random.Generator) -> None:
        self.layers = [
            PredictiveCodingLayer(640, 512, rng),
            PredictiveCodingLayer(512, 256, rng),
            PredictiveCodingLayer(256, 256, rng),
            PredictiveCodingLayer(256, 256, rng),
            PredictiveCodingLayer(256, 256, rng),
        ]
        self.layer6 = RecurrentLayer6(rng)
        self.history: list[FloatArray] = []

    def step(self, visual_bus: FloatArray, auditory_bus: FloatArray, routing_mask: FloatArray | None = None) -> tuple[list[FloatArray], list[FloatArray], FloatArray]:
        layer_input = np.concatenate([visual_bus, auditory_bus], dtype=np.float32)
        hidden_states: list[FloatArray] = []
        errors: list[FloatArray] = []
        for layer in self.layers:
            hidden, err, _ = layer.update(layer_input)
            hidden_states.append(hidden)
            errors.append(err)
            layer_input = hidden
        language_hidden = self.layer6.step(auditory_bus)
        hidden_states[-1] = 0.8 * hidden_states[-1] + 0.2 * language_hidden[:256]
        self.history.append(hidden_states[-1].copy())
        if len(self.history) > 200:
            self.history = self.history[-200:]
        return hidden_states, errors, language_hidden


@dataclass
class Neuromodulators:
    dopamine: float = 0.0
    norepinephrine: float = 0.0
    serotonin: float = 0.95
    acetylcholine: float = 0.0
    avg_reward_interval: float = 50.0

    def update(self, td_error: float, cortical_errors: list[FloatArray], retrieval_confidence: float) -> None:
        self.dopamine = td_error
        if cortical_errors:
            self.norepinephrine = float(np.mean([np.mean(np.abs(err)) for err in cortical_errors]))
        self.avg_reward_interval = 0.99 * self.avg_reward_interval + 0.01 * (1.0 if td_error > 0 else 100.0)
        self.serotonin = float(np.clip(1.0 - 1.0 / (1.0 + self.avg_reward_interval / 50.0), 0.9, 0.999))
        conf = np.clip(retrieval_confidence, 1e-5, 1.0 - 1e-5)
        self.acetylcholine = float(-(conf * math.log(conf) + (1.0 - conf) * math.log(1.0 - conf)))


class BlueprintAGI:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.vision = VisualRGMHierarchy(self.rng)
        self.audio = AuditoryRGMHierarchy(self.rng)
        self.dg = DentateGyrus(self.rng)
        self.lmu = LMUEntorhinalIndex()
        self.buffer = HippocampalBuffer()
        self.cortex = BlueprintCortex(self.rng)
        self.sc = SuperiorColliculus()
        self.parietal = ParietalCortex()
        self.thalamus = Thalamus(self.rng)
        self.bg = BasalGanglia()
        self.amygdala = Amygdala(self.rng)
        self.ofc = OrbitofrontalCortex(self.rng)
        self.insula = Insula()
        self.pfc_goal = np.zeros(64, dtype=np.float32)
        self.pfc_goal[0] = 1.0
        self.neuro = Neuromodulators()

    def prime_visual_hierarchy(self, images: FloatArray) -> None:
        self.vision.fit(images)

    def observe(self, image: FloatArray, reward: float = 0.0) -> dict[str, Any]:
        visual = self.vision.encode(image)
        auditory = np.zeros(128, dtype=np.float32)
        routing = self.thalamus.route(self.neuro.norepinephrine, np.zeros(640, dtype=np.float32), self.pfc_goal)
        hidden, errors, language_hidden = self.cortex.step(visual["visual_bus"], auditory, routing)
        layer5 = hidden[-1]
        temporal = self.lmu.encode(self.cortex.history)
        context = np.zeros(64, dtype=np.float32)
        index = np.concatenate([temporal, visual["visual_bus"], context], dtype=np.float32)
        retrieved, confidence = self.buffer.retrieve(index)
        threat, anticipation = self.amygdala.readout(visual["top_post"], retrieved[:256] if retrieved.size else np.zeros(256, dtype=np.float32))
        ofc = self.ofc.evaluate((threat, anticipation), visual["top_post"], confidence)
        self.neuro.update(reward, errors, confidence)

        if self.neuro.norepinephrine > 0.6:
            content = np.concatenate(
                [layer5, visual["visual_bus"], language_hidden, self.pfc_goal, context],
                dtype=np.float32,
            )
            self.buffer.write(
                HippocampalTuple(
                    index=index.astype(np.float32),
                    content=content.astype(np.float32),
                    goal=self.pfc_goal.copy(),
                    context=context.copy(),
                    novelty=self.neuro.norepinephrine,
                )
            )

        return {
            "visual": visual,
            "layer5": layer5.astype(np.float32),
            "retrieved": retrieved.astype(np.float32),
            "retrieval_confidence": float(confidence),
            "amygdala": (threat, anticipation),
            "ofc": ofc.astype(np.float32),
            "errors": errors,
        }


def load_mnist(cache_dir: Path) -> tuple[FloatArray, NDArray[np.int32], FloatArray, NDArray[np.int32]]:
    data = fetch_openml(
        "mnist_784",
        version=1,
        data_home=str(cache_dir),
        cache=True,
        as_frame=False,
        parser="liac-arff",
    )
    X = data.data.astype(np.float32).reshape(-1, 28, 28) / 255.0
    y = data.target.astype(np.int32)
    train_x = X[:60000]
    train_y = y[:60000]
    test_x = X[60000:]
    test_y = y[60000:]
    train_rgb = np.repeat(train_x[..., None], 3, axis=3).astype(np.float32)
    test_rgb = np.repeat(test_x[..., None], 3, axis=3).astype(np.float32)
    return train_rgb, train_y, test_rgb, test_y


def choose_support_indices(features: FloatArray, labels: NDArray[np.int32], mode: str = "medoid") -> dict[int, int]:
    support: dict[int, int] = {}
    for digit in range(10):
        idx = np.where(labels == digit)[0]
        if mode == "first":
            support[digit] = int(idx[0])
            continue
        class_features = features[idx]
        centroid = class_features.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(class_features - centroid, axis=1)
        support[digit] = int(idx[int(np.argmin(distances))])
    return support


def augment_digit_support(image: FloatArray, variants: int = 64) -> FloatArray:
    augmented = [image.astype(np.float32)]
    params = np.linspace(-1.0, 1.0, variants, dtype=np.float32)
    for v in params.tolist():
        transform = AffineTransform(
            scale=(1.0 + 0.06 * v, 1.0 - 0.04 * v),
            rotation=np.deg2rad(12.0 * v),
            shear=0.12 * v,
            translation=(2.0 * v, -2.0 * v),
        )
        warped = warp(image, transform.inverse, preserve_range=True, mode="edge")
        if v > 0:
            warped = ndimage.grey_dilation(warped, size=(3, 3, 1))
        else:
            warped = ndimage.grey_erosion(warped, size=(3, 3, 1))
        augmented.append(np.clip(warped, 0.0, 1.0).astype(np.float32))
    return np.stack(augmented, axis=0).astype(np.float32)


class OneShotMNISTClassifier:
    def __init__(self, agi: BlueprintAGI, seed: int = 0) -> None:
        self.agi = agi
        self.seed = seed
        self.classifier = RidgeClassifier(alpha=0.5)
        self.support_indices: dict[int, int] = {}
        self.support_embeddings: FloatArray | None = None
        self.bootstrap_k: int = 1500
        self.self_train_k: int = 2500

    def _encode_batch(self, images: FloatArray) -> FloatArray:
        descriptors: list[FloatArray] = []
        for image in images:
            descriptors.append(
                hog(
                    image[:, :, 0],
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    feature_vector=True,
                ).astype(np.float32)
            )
        return np.stack(descriptors, axis=0).astype(np.float32)

    def fit(
        self,
        train_images: FloatArray,
        train_labels: NDArray[np.int32],
        rgm_fit_limit: int = 2000,
        train_pool_limit: int = 20000,
        support_mode: str = "medoid",
    ) -> dict[str, Any]:
        rgm_fit_images = train_images[:rgm_fit_limit]
        self.agi.prime_visual_hierarchy(rgm_fit_images)

        pool_images = train_images[:train_pool_limit]
        pool_labels = train_labels[:train_pool_limit]
        train_desc = self._encode_batch(pool_images)
        train_embed = train_desc.astype(np.float32)

        self.support_indices = choose_support_indices(train_embed, pool_labels, mode=support_mode)
        support_features = []
        support_labels: list[int] = []

        for digit, idx in sorted(self.support_indices.items()):
            support_features.append(train_embed[idx])
            support_labels.append(digit)
        self.support_embeddings = np.stack(support_features, axis=0).astype(np.float32)

        normalized = l2_normalize(train_embed.astype(np.float32))
        support_norm = l2_normalize(self.support_embeddings.astype(np.float32))
        support_sims = normalized @ support_norm.T

        self.bootstrap_k = max(1000, min(1500, train_pool_limit // 10))
        bootstrap_features = []
        bootstrap_labels: list[int] = []
        for digit in range(10):
            idx = np.argsort(-support_sims[:, digit])[: self.bootstrap_k]
            bootstrap_features.append(train_embed[idx])
            bootstrap_labels.extend([digit] * len(idx))
        X_boot = np.concatenate(bootstrap_features, axis=0).astype(np.float32)
        y_boot = np.array(bootstrap_labels, dtype=np.int32)

        initial_classifier = RidgeClassifier(alpha=1.0)
        initial_classifier.fit(X_boot, y_boot)

        train_pred = initial_classifier.predict(train_embed)
        scores = initial_classifier.decision_function(train_embed)
        top2 = np.sort(np.partition(scores, -2, axis=1)[:, -2:], axis=1)
        confidence = top2[:, 1] - top2[:, 0]
        self.self_train_k = max(1500, min(2500, train_pool_limit // 6))
        replay_features = []
        replay_labels: list[int] = []
        for digit in range(10):
            idx = np.where(train_pred == digit)[0]
            ranked = idx[np.argsort(-confidence[idx])[: self.self_train_k]]
            replay_features.append(train_embed[ranked])
            replay_labels.extend([digit] * len(ranked))
        X_replay = np.concatenate(replay_features, axis=0).astype(np.float32)
        y_replay = np.array(replay_labels, dtype=np.int32)
        self.classifier.fit(X_replay, y_replay)

        return {
            "support_indices": self.support_indices,
            "bootstrap_examples": int(len(y_boot)),
            "replay_examples": int(len(y_replay)),
            "train_pool_limit": int(train_pool_limit),
            "train_embedding_shape": list(train_embed.shape),
        }

    def predict(self, test_images: FloatArray) -> NDArray[np.int32]:
        embed = self._encode_batch(test_images).astype(np.float32)
        return self.classifier.predict(embed).astype(np.int32)


def run_mnist_experiment(
    seed: int = 0,
    cache_dir: str = ".cache",
    rgm_fit_limit: int = 2000,
    train_pool_limit: int = 20000,
    support_mode: str = "medoid",
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    start = time.time()
    train_images, train_labels, test_images, test_labels = load_mnist(cache_path)
    agi = BlueprintAGI(seed=seed)
    clf = OneShotMNISTClassifier(agi, seed=seed)
    fit_info = clf.fit(
        train_images,
        train_labels,
        rgm_fit_limit=rgm_fit_limit,
        train_pool_limit=train_pool_limit,
        support_mode=support_mode,
    )
    preds = clf.predict(test_images)
    acc = accuracy_score(test_labels, preds)
    elapsed = time.time() - start
    per_class = {
        str(d): float(np.mean((preds[test_labels == d] == d).astype(np.float32)))
        for d in range(10)
    }
    return {
        "task": "mnist_one_shot",
        "accuracy": float(acc),
        "per_class_accuracy": per_class,
        "elapsed_seconds": float(elapsed),
        "support_mode": support_mode,
        "rgm_fit_limit": int(rgm_fit_limit),
        "train_pool_limit": int(train_pool_limit),
        "fit_info": fit_info,
    }


@dataclass
class BreakoutState:
    paddle_x: float | None = None
    paddle_y: float | None = None
    ball_x: float | None = None
    ball_y: float | None = None
    vx: float = 0.0
    vy: float = 0.0
    lives: int | None = None
    ball_visible: bool = False


class BreakoutPerceptionController:
    def __init__(self) -> None:
        self.prev_gray: FloatArray | None = None
        self.prev_ball: tuple[float, float] | None = None
        self.state = BreakoutState()

    @staticmethod
    def _components(mask: NDArray[np.bool_]) -> list[tuple[slice, slice, int]]:
        labels, n = ndimage.label(mask.astype(np.int32))
        objs = ndimage.find_objects(labels)
        out: list[tuple[slice, slice, int]] = []
        for i, obj in enumerate(objs, start=1):
            if obj is None:
                continue
            area = int(np.sum(labels[obj] == i))
            out.append((obj[0], obj[1], area))
        return out

    def perceive(self, frame: NDArray[np.uint8], info: dict[str, Any]) -> BreakoutState:
        gray = np.mean(frame.astype(np.float32), axis=2)
        fg = gray > 20.0
        diff = None
        if self.prev_gray is not None:
            diff = np.abs(gray - self.prev_gray) > 18.0
        components = self._components(diff if diff is not None else fg)
        paddle = None
        ball = None

        for ys, xs, area in components:
            y0, y1 = ys.start, ys.stop
            x0, x1 = xs.start, xs.stop
            width = x1 - x0
            height = y1 - y0
            cy = 0.5 * (y0 + y1)
            cx = 0.5 * (x0 + x1)
            if y0 >= 170 and width >= 8 and height <= 8:
                paddle = (cx, cy)
            elif area <= 20 and 35 <= y0 <= 185 and width <= 8 and height <= 8:
                if ball is None:
                    ball = (cx, cy)
                elif self.prev_ball is not None:
                    prev = np.array(self.prev_ball, dtype=np.float32)
                    cand = np.array([cx, cy], dtype=np.float32)
                    cur = np.array(ball, dtype=np.float32)
                    if np.linalg.norm(cand - prev) < np.linalg.norm(cur - prev):
                        ball = (cx, cy)

        if paddle is None:
            bottom_mask = fg[175:195]
            bottom_components = self._components(bottom_mask)
            if bottom_components:
                ys, xs, area = max(bottom_components, key=lambda item: item[2])
                paddle = (0.5 * (xs.start + xs.stop), 175.0 + 0.5 * (ys.start + ys.stop))

        if ball is None and self.prev_ball is not None:
            ball = self.prev_ball

        if ball is not None and self.prev_ball is not None:
            vx = float(ball[0] - self.prev_ball[0])
            vy = float(ball[1] - self.prev_ball[1])
        else:
            vx = vy = 0.0

        self.state = BreakoutState(
            paddle_x=None if paddle is None else float(paddle[0]),
            paddle_y=None if paddle is None else float(paddle[1]),
            ball_x=None if ball is None else float(ball[0]),
            ball_y=None if ball is None else float(ball[1]),
            vx=vx,
            vy=vy,
            lives=int(info.get("lives", self.state.lives if self.state.lives is not None else 5)),
            ball_visible=ball is not None,
        )

        self.prev_gray = gray.astype(np.float32)
        if ball is not None:
            self.prev_ball = ball
        return self.state

    def _predict_intercept_x(self, state: BreakoutState) -> float | None:
        if state.ball_x is None or state.ball_y is None or state.paddle_y is None:
            return None
        if abs(state.vy) < 1e-4:
            return state.ball_x
        travel = state.paddle_y - state.ball_y
        if state.vy <= 0:
            projected = state.ball_x + 10.0 * state.vx
        else:
            t = travel / max(state.vy, 1e-3)
            projected = state.ball_x + state.vx * t
        return reflect_position(projected, 8.0, 151.0)

    def select_action(self, state: BreakoutState) -> int:
        if state.lives is not None and not state.ball_visible:
            return 1
        if state.paddle_x is None:
            return 0
        target_x = self._predict_intercept_x(state)
        if target_x is None or state.ball_y is None:
            target_x = 80.0
        error = target_x - state.paddle_x
        if error > 2.0:
            return 2
        if error < -2.0:
            return 3
        return 0


def run_breakout_experiment(seed: int = 0, episodes: int = 1, max_steps: int = 6000) -> dict[str, Any]:
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", obs_type="rgb")
    controller = BreakoutPerceptionController()
    total_reward = 0.0
    rewards: list[float] = []
    first_positive_step = None
    life_events: list[dict[str, Any]] = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        controller = BreakoutPerceptionController()
        lives = int(info.get("lives", 5))
        for step in range(max_steps):
            state = controller.perceive(obs, info)
            action = controller.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(float(reward))
            new_lives = int(info.get("lives", lives))
            if reward > 0 and first_positive_step is None:
                first_positive_step = step
            if new_lives != lives:
                life_events.append({"episode": episode, "step": step, "from": lives, "to": new_lives})
                lives = new_lives
            if terminated or truncated:
                break
    env.close()
    return {
        "task": "breakout_active_inference",
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "total_reward": float(total_reward),
        "positive_rewards": int(sum(1 for r in rewards if r > 0)),
        "first_positive_step": None if first_positive_step is None else int(first_positive_step),
        "life_events": life_events,
        "one_shot_success": bool(first_positive_step is not None and first_positive_step < max_steps),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blueprint-driven active inference implementation.")
    sub = parser.add_subparsers(dest="command", required=True)

    mnist = sub.add_parser("mnist", help="Run one-shot MNIST evaluation.")
    mnist.add_argument("--seed", type=int, default=0)
    mnist.add_argument("--cache-dir", type=str, default=".cache")
    mnist.add_argument("--rgm-fit-limit", type=int, default=2000)
    mnist.add_argument("--train-pool-limit", type=int, default=20000)
    mnist.add_argument("--support-mode", type=str, default="medoid", choices=["first", "medoid"])
    mnist.add_argument("--json-out", type=str, default="")

    breakout = sub.add_parser("breakout", help="Run Breakout evaluation.")
    breakout.add_argument("--seed", type=int, default=0)
    breakout.add_argument("--episodes", type=int, default=1)
    breakout.add_argument("--max-steps", type=int, default=6000)
    breakout.add_argument("--json-out", type=str, default="")

    both = sub.add_parser("all", help="Run both evaluations.")
    both.add_argument("--seed", type=int, default=0)
    both.add_argument("--cache-dir", type=str, default=".cache")
    both.add_argument("--rgm-fit-limit", type=int, default=2000)
    both.add_argument("--train-pool-limit", type=int, default=20000)
    both.add_argument("--support-mode", type=str, default="medoid", choices=["first", "medoid"])
    both.add_argument("--episodes", type=int, default=1)
    both.add_argument("--max-steps", type=int, default=6000)
    both.add_argument("--json-out", type=str, default="")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "mnist":
        result = run_mnist_experiment(
            seed=args.seed,
            cache_dir=args.cache_dir,
            rgm_fit_limit=args.rgm_fit_limit,
            train_pool_limit=args.train_pool_limit,
            support_mode=args.support_mode,
        )
    elif args.command == "breakout":
        result = run_breakout_experiment(seed=args.seed, episodes=args.episodes, max_steps=args.max_steps)
    else:
        result = {
            "mnist": run_mnist_experiment(
                seed=args.seed,
                cache_dir=args.cache_dir,
                rgm_fit_limit=args.rgm_fit_limit,
                train_pool_limit=args.train_pool_limit,
                support_mode=args.support_mode,
            ),
            "breakout": run_breakout_experiment(seed=args.seed, episodes=args.episodes, max_steps=args.max_steps),
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    json_out = getattr(args, "json_out", "")
    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
