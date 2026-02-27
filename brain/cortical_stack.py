"""Cortical stack with retina, cortex, temporal, hippocampal, and control modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Retina:
    """Center-surround retinal preprocessing."""

    def encode(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.ndim == 1:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
            padded = np.pad(obs, (1, 1), mode="edge")
            local = np.convolve(padded, kernel, mode="valid")
            return obs - local

        if obs.ndim == 2:
            padded = np.pad(obs, ((1, 1), (1, 1)), mode="edge")
            local = (
                padded[:-2, :-2]
                + padded[1:-1, :-2]
                + padded[2:, :-2]
                + padded[:-2, 1:-1]
                + padded[1:-1, 1:-1]
                + padded[2:, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, 2:]
                + padded[2:, 2:]
            ) / 9.0
            return (obs - local).ravel()

        return obs.ravel()


@dataclass
class VisualCortex:
    """Sparse visual feature extractor."""

    input_dim: int
    latent_dim: int = 64
    sparsity: float = 0.2
    learning_rate: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.dictionary = rng.normal(0.0, 0.15, size=(self.latent_dim, self.input_dim))

    def encode(self, retinal: np.ndarray, learn: bool = False) -> np.ndarray:
        retinal = np.asarray(retinal, dtype=np.float64).ravel()
        if retinal.shape[0] != self.input_dim:
            self._resize(retinal.shape[0])

        activation = np.tanh(self.dictionary @ retinal)
        k = max(1, int(self.sparsity * activation.size))
        top_indices = np.argpartition(np.abs(activation), -k)[-k:]
        sparse = np.zeros_like(activation)
        sparse[top_indices] = activation[top_indices]

        if learn:
            reconstruction_error = retinal - self.dictionary.T @ sparse
            self.dictionary += self.learning_rate * np.outer(sparse, reconstruction_error)

        return sparse

    def _resize(self, input_dim: int) -> None:
        rng = np.random.default_rng(input_dim + self.latent_dim)
        self.input_dim = input_dim
        self.dictionary = rng.normal(0.0, 0.15, size=(self.latent_dim, self.input_dim))


@dataclass
class AnteriorTemporalLobe:
    """Semantic binder through adaptive prototype memory."""

    feature_dim: int
    slots: int = 24
    decay: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.prototypes = rng.normal(0.0, 0.1, size=(self.slots, self.feature_dim))

    def bind(self, features: np.ndarray, learn: bool = False) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64).ravel()
        if features.shape[0] != self.feature_dim:
            raise ValueError("AnteriorTemporalLobe input dimension mismatch.")

        scores = self.prototypes @ features
        weights = np.exp(scores - np.max(scores))
        weights /= np.sum(weights) + 1e-12
        semantic = np.sum(weights[:, None] * self.prototypes, axis=0)

        if learn:
            winner = int(np.argmax(weights))
            self.prototypes[winner] = self.decay * self.prototypes[winner] + (1.0 - self.decay) * features

        return semantic


@dataclass
class Hippocampus:
    """Episodic memory with nearest-neighbor recall."""

    feature_dim: int
    capacity: int = 1024

    def __post_init__(self) -> None:
        self.memory: list[np.ndarray] = []

    def store(self, features: np.ndarray) -> None:
        vector = np.asarray(features, dtype=np.float64).ravel()
        self.memory.append(vector)
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity :]

    def recall(self, cue: np.ndarray, k: int = 4) -> np.ndarray:
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if not self.memory:
            return np.zeros(self.feature_dim, dtype=np.float64)

        memory = np.asarray(self.memory)
        distances = np.linalg.norm(memory - cue[None, :], axis=1)
        nearest = np.argsort(distances)[: max(1, min(k, len(self.memory)))]
        weights = np.exp(-distances[nearest])
        weights /= np.sum(weights) + 1e-12
        return np.sum(memory[nearest] * weights[:, None], axis=0)


@dataclass
class Homeostasis:
    """Tracks internal regulation variables for active control."""

    target_energy: float = 1.0
    target_stability: float = 1.0
    adaptation: float = 0.05

    def __post_init__(self) -> None:
        self.energy = self.target_energy
        self.stability = self.target_stability

    def update(self, reward: float, prediction_error: float) -> None:
        reward_signal = np.tanh(float(reward))
        surprise = np.tanh(float(prediction_error))
        self.energy = np.clip(self.energy + self.adaptation * reward_signal, 0.0, 2.0)
        self.stability = np.clip(self.stability - self.adaptation * surprise, 0.0, 2.0)

    def error(self) -> float:
        energy_error = abs(self.target_energy - self.energy)
        stability_error = abs(self.target_stability - self.stability)
        return float(0.5 * (energy_error + stability_error))


@dataclass
class PFCVLPFC:
    """Executive temperature modulation from free-energy pressure."""

    base_temperature: float = 1.0
    min_temperature: float = 0.15
    max_temperature: float = 2.5
    gain: float = 0.8
    temperature: float = 1.0

    def update(self, free_energy: float, prediction_error: float) -> float:
        signal = float(free_energy + prediction_error)
        control = 1.0 + self.gain * np.tanh(signal)
        self.temperature = float(
            np.clip(self.base_temperature * control, self.min_temperature, self.max_temperature)
        )
        return self.temperature


@dataclass
class CorticalStack:
    """Integrated cortical processing pipeline."""

    observation_dim: int
    visual_latent_dim: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        self.retina = Retina()
        self.visual = VisualCortex(self.observation_dim, latent_dim=self.visual_latent_dim, seed=self.seed)
        self.atl = AnteriorTemporalLobe(self.visual_latent_dim, seed=self.seed + 13)
        self.hippocampus = Hippocampus(self.visual_latent_dim)
        self.homeostasis = Homeostasis()
        self.pfc_vlpfc = PFCVLPFC()

    def process(self, observation: np.ndarray, learn: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        retinal = self.retina.encode(observation)
        visual = self.visual.encode(retinal, learn=learn)
        semantic = self.atl.bind(visual, learn=learn)
        recall = self.hippocampus.recall(visual)
        combined = np.concatenate([visual, semantic, recall], axis=0)

        if learn:
            self.hippocampus.store(visual)

        diagnostics = {
            "retina": retinal,
            "visual": visual,
            "semantic": semantic,
            "recall": recall,
        }
        return combined, diagnostics
