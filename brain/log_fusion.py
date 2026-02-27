"""Log-space Bayesian fusion with exponential-conjugate updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.temporal_decay import MarkovTemporalDecay


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value)) + 1e-12))


@dataclass
class ExponentialConjugateComponent:
    """Diagonal Gaussian component represented in natural parameter form."""

    dim: int
    prior_precision: float = 1.0
    obs_precision: float = 4.0
    ridge: float = 1e-4
    eta1: np.ndarray = field(init=False)
    eta2: np.ndarray = field(init=False)
    count: float = field(default=1.0)

    def __post_init__(self) -> None:
        precision = np.full(self.dim, self.prior_precision, dtype=np.float64)
        self.eta1 = np.zeros(self.dim, dtype=np.float64)
        self.eta2 = -0.5 * precision

    @property
    def mean(self) -> np.ndarray:
        precision = self.precision
        return self.eta1 / (precision + self.ridge)

    @property
    def precision(self) -> np.ndarray:
        return np.maximum(-2.0 * self.eta2, self.ridge)

    @property
    def variance(self) -> np.ndarray:
        return 1.0 / (self.precision + self.ridge)

    def warm_start(self, observation: np.ndarray) -> None:
        observation = np.asarray(observation, dtype=np.float64)
        precision = np.full(self.dim, self.obs_precision, dtype=np.float64)
        self.eta1 = precision * observation
        self.eta2 = -0.5 * precision
        self.count = 1.0

    def log_likelihood(self, observation: np.ndarray) -> float:
        observation = np.asarray(observation, dtype=np.float64)
        delta = observation - self.mean
        variance = self.variance
        term = np.sum((delta**2) / (variance + self.ridge) + np.log(2.0 * np.pi * variance + self.ridge))
        return -0.5 * float(term)

    def update(self, observation: np.ndarray, responsibility: float, decay: MarkovTemporalDecay) -> None:
        responsibility = float(np.clip(responsibility, 0.0, 1.0))
        if responsibility <= 0.0:
            return

        observation = np.asarray(observation, dtype=np.float64)
        contribution_precision = self.obs_precision * responsibility
        target_eta1 = self.eta1 + contribution_precision * observation
        target_eta2 = self.eta2 - 0.5 * contribution_precision

        self.eta1 = decay.blend(self.eta1, target_eta1)
        self.eta2 = decay.blend(self.eta2, target_eta2)
        self.count = decay.blend_scalar(self.count, self.count + responsibility)

    def absorb(self, other: "ExponentialConjugateComponent", self_weight: float, other_weight: float) -> None:
        total = self_weight + other_weight + 1e-12
        alpha = self_weight / total
        beta = other_weight / total
        self.eta1 = alpha * self.eta1 + beta * other.eta1
        self.eta2 = alpha * self.eta2 + beta * other.eta2
        self.count = alpha * self.count + beta * other.count


@dataclass
class LogSpaceFusion:
    """Adaptive mixture with log-space fusion and component growth."""

    dim: int
    initial_components: int = 2
    max_components: int = 24
    growth_surprise_threshold: float = 6.0
    decay: MarkovTemporalDecay = field(default_factory=MarkovTemporalDecay)
    components: list[ExponentialConjugateComponent] = field(init=False)
    log_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.components = [ExponentialConjugateComponent(self.dim) for _ in range(self.initial_components)]
        self.log_weights = np.full(self.initial_components, -np.log(self.initial_components), dtype=np.float64)

    def _log_joint(self, observation: np.ndarray) -> np.ndarray:
        likelihoods = np.array([component.log_likelihood(observation) for component in self.components], dtype=np.float64)
        return self.log_weights + likelihoods

    def posterior(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        log_joint = self._log_joint(observation)
        log_norm = _logsumexp(log_joint)
        responsibilities = np.exp(log_joint - log_norm)
        surprise = -log_norm
        return responsibilities, float(surprise)

    def update(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        observation = np.asarray(observation, dtype=np.float64)
        responsibilities, surprise = self.posterior(observation)

        for component, resp in zip(self.components, responsibilities):
            component.update(observation, float(resp), self.decay)

        log_resp = np.log(responsibilities + 1e-12)
        blended = self.log_weights + log_resp
        self.log_weights = blended - _logsumexp(blended)

        if surprise > self.growth_surprise_threshold and len(self.components) < self.max_components:
            self._grow(observation)

        return responsibilities, surprise

    def _grow(self, observation: np.ndarray) -> None:
        new_component = ExponentialConjugateComponent(self.dim)
        new_component.warm_start(observation)
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
