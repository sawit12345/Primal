"""Slot-centric GMM fusion with log-space Bayesian updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.temporal_decay import MarkovTemporalDecay


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value)) + 1e-12))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / (np.sum(values) + 1e-12)


@dataclass
class ExponentialConjugateComponent:
    """Diagonal Gaussian component with object-slot affinity."""

    dim: int
    slot_count: int
    ridge: float = 1e-4
    initial_variance: float = 1.0
    mean: np.ndarray = field(init=False)
    variance: np.ndarray = field(init=False)
    count: float = field(default=1e-3)
    slot_logits: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.variance = np.full(self.dim, self.initial_variance, dtype=np.float64)
        self.slot_logits = np.zeros(self.slot_count, dtype=np.float64)

    @property
    def slot_distribution(self) -> np.ndarray:
        return _softmax(self.slot_logits)

    def warm_start(self, observation: np.ndarray, slot_probs: np.ndarray) -> None:
        observation = np.asarray(observation, dtype=np.float64)
        self.mean = observation.copy()
        self.variance = np.full(self.dim, 0.5, dtype=np.float64)
        self.count = 1.0
        self.slot_logits = np.log(np.asarray(slot_probs, dtype=np.float64) + 1e-8)

    def log_likelihood(self, observation: np.ndarray) -> float:
        observation = np.asarray(observation, dtype=np.float64)
        delta = observation - self.mean
        variance = np.maximum(self.variance, self.ridge)
        term = np.sum((delta**2) / (variance + self.ridge) + np.log(2.0 * np.pi * variance + self.ridge))
        return -0.5 * float(term)

    def update(
        self,
        observation: np.ndarray,
        responsibility: float,
        slot_probs: np.ndarray,
        decay: MarkovTemporalDecay,
    ) -> None:
        responsibility = float(np.clip(responsibility, 0.0, 1.0))
        if responsibility <= 0.0:
            return

        observation = np.asarray(observation, dtype=np.float64)
        target_count = self.count + responsibility
        step = responsibility / (target_count + self.ridge)

        mean_delta = observation - self.mean
        target_mean = self.mean + step * mean_delta
        target_variance = (1.0 - step) * self.variance + step * (mean_delta**2)
        target_variance = np.clip(target_variance, 1e-4, 1e4)

        target_slot_logits = np.log(np.asarray(slot_probs, dtype=np.float64) + 1e-8)

        self.mean = decay.blend(self.mean, target_mean)
        self.variance = decay.blend(self.variance, target_variance)
        self.slot_logits = decay.blend(self.slot_logits, target_slot_logits)
        self.count = decay.blend_scalar(self.count, target_count)

    def absorb(self, other: "ExponentialConjugateComponent", self_weight: float, other_weight: float) -> None:
        total = self_weight + other_weight + 1e-12
        alpha = self_weight / total
        beta = other_weight / total

        mean_a = self.mean.copy()
        mean_b = other.mean
        self.mean = alpha * mean_a + beta * mean_b
        between_var = alpha * beta * (mean_a - mean_b) ** 2
        self.variance = np.clip(alpha * self.variance + beta * other.variance + between_var, 1e-4, 1e4)
        self.slot_logits = alpha * self.slot_logits + beta * other.slot_logits
        self.count = alpha * self.count + beta * other.count


@dataclass
class LogSpaceFusion:
    """Adaptive slot-centric GMM with online component growth."""

    dim: int
    slot_count: int = 8
    initial_components: int = 2
    max_components: int = 24
    growth_surprise_threshold: float = 6.0
    slot_temperature: float = 0.8
    seed: int = 0
    decay: MarkovTemporalDecay = field(default_factory=MarkovTemporalDecay)
    components: list[ExponentialConjugateComponent] = field(init=False)
    log_weights: np.ndarray = field(init=False)
    slot_projector: np.ndarray = field(init=False)
    slot_prototypes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.slot_projector = rng.normal(0.0, 1.0 / np.sqrt(self.dim), size=(self.slot_count, self.dim))
        self.slot_prototypes = np.zeros((self.slot_count, self.dim), dtype=np.float64)

        self.components = [
            ExponentialConjugateComponent(self.dim, slot_count=self.slot_count)
            for _ in range(self.initial_components)
        ]
        self.log_weights = np.full(self.initial_components, -np.log(self.initial_components), dtype=np.float64)

    def slot_attention(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64).ravel()
        normalized = observation / (np.linalg.norm(observation) + 1e-8)
        projected = self.slot_projector @ normalized
        prototype_match = self.slot_prototypes @ normalized
        logits = projected + prototype_match
        return _softmax(logits / max(self.slot_temperature, 1e-3))

    def _log_joint(self, observation: np.ndarray, slot_probs: np.ndarray) -> np.ndarray:
        log_joint = []
        for idx, component in enumerate(self.components):
            slot_match = float(np.dot(slot_probs, component.slot_distribution))
            log_joint.append(
                self.log_weights[idx] + component.log_likelihood(observation) + np.log(slot_match + 1e-8)
            )
        return np.asarray(log_joint, dtype=np.float64)

    def posterior(self, observation: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        slot_probs = self.slot_attention(observation)
        log_joint = self._log_joint(observation, slot_probs)
        log_norm = _logsumexp(log_joint)
        responsibilities = np.exp(log_joint - log_norm)
        surprise = -log_norm
        return responsibilities, float(surprise), slot_probs

    def update(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        observation = np.asarray(observation, dtype=np.float64)
        responsibilities, surprise, slot_probs = self.posterior(observation)

        for component, resp in zip(self.components, responsibilities):
            component.update(observation, float(resp), slot_probs, self.decay)

        log_resp = np.log(responsibilities + 1e-12)
        blended = self.decay.blend(self.log_weights, log_resp)
        self.log_weights = blended - _logsumexp(blended)

        target_slot_prototypes = np.outer(slot_probs, observation)
        self.slot_prototypes = self.decay.blend(self.slot_prototypes, target_slot_prototypes)
        norms = np.linalg.norm(self.slot_prototypes, axis=1, keepdims=True) + 1e-8
        self.slot_prototypes = self.slot_prototypes / norms

        if surprise > self.growth_surprise_threshold and len(self.components) < self.max_components:
            self._grow(observation, slot_probs)

        return responsibilities, surprise

    def _grow(self, observation: np.ndarray, slot_probs: np.ndarray) -> None:
        new_component = ExponentialConjugateComponent(self.dim, slot_count=self.slot_count)
        new_component.warm_start(observation, slot_probs)
        self.components.append(new_component)
        new_log_weights = np.append(self.log_weights, np.log(1e-2))
        self.log_weights = new_log_weights - _logsumexp(new_log_weights)

    def predictive_mean(self) -> np.ndarray:
        weights = np.exp(self.log_weights)
        means = np.asarray([component.mean for component in self.components])
        return np.sum(weights[:, None] * means, axis=0)

    def predictive_variance(self) -> np.ndarray:
        weights = np.exp(self.log_weights)
        variances = np.asarray([component.variance for component in self.components])
        return np.sum(weights[:, None] * variances, axis=0)

    def slot_affinity(self, first: np.ndarray, second: np.ndarray) -> float:
        first_slots = self.slot_attention(first)
        second_slots = self.slot_attention(second)
        return float(np.dot(first_slots, second_slots))
