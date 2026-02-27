"""Bayesian model reduction for component pruning and merging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.log_fusion import LogSpaceFusion


def _kl_diag_gaussian(mean_a: np.ndarray, var_a: np.ndarray, mean_b: np.ndarray, var_b: np.ndarray) -> float:
    ratio = (var_a + 1e-8) / (var_b + 1e-8)
    delta = (mean_b - mean_a) ** 2 / (var_b + 1e-8)
    dim = mean_a.shape[0]
    return 0.5 * float(np.sum(ratio + delta - 1.0 - np.log(ratio + 1e-8)) / max(1, dim))


@dataclass
class BayesianModelReduction:
    """Prunes weak components and merges redundant components."""

    prune_log_weight: float = -8.0
    merge_kl_threshold: float = 0.08
    max_components: int = 12

    def reduce(self, fusion: LogSpaceFusion) -> None:
        if len(fusion.components) <= 1:
            return

        self._prune(fusion)
        self._merge(fusion)
        self._enforce_max_components(fusion)

    def _prune(self, fusion: LogSpaceFusion) -> None:
        keep_indices = [i for i, weight in enumerate(fusion.log_weights) if weight >= self.prune_log_weight]
        if not keep_indices:
            keep_indices = [int(np.argmax(fusion.log_weights))]
        self._select_indices(fusion, keep_indices)

    def _merge(self, fusion: LogSpaceFusion) -> None:
        changed = True
        while changed and len(fusion.components) > 1:
            changed = False
            weights = np.exp(fusion.log_weights)
            best_pair: tuple[int, int] | None = None
            best_kl = np.inf

            for i in range(len(fusion.components)):
                for j in range(i + 1, len(fusion.components)):
                    component_i = fusion.components[i]
                    component_j = fusion.components[j]
                    kl = _kl_diag_gaussian(component_i.mean, component_i.variance, component_j.mean, component_j.variance)
                    if kl < best_kl:
                        best_kl = kl
                        best_pair = (i, j)

            if best_pair is not None and best_kl <= self.merge_kl_threshold:
                i, j = best_pair
                first = fusion.components[i]
                second = fusion.components[j]
                first.absorb(second, weights[i], weights[j])
                del fusion.components[j]

                log_weight_i = np.log(weights[i] + weights[j] + 1e-12)
                fusion.log_weights = np.delete(fusion.log_weights, j)
                fusion.log_weights[i] = log_weight_i
                fusion.log_weights -= np.log(np.sum(np.exp(fusion.log_weights)) + 1e-12)
                changed = True

    def _enforce_max_components(self, fusion: LogSpaceFusion) -> None:
        if len(fusion.components) <= self.max_components:
            return
        sorted_indices = np.argsort(fusion.log_weights)[::-1][: self.max_components]
        self._select_indices(fusion, sorted_indices.tolist())

    def _select_indices(self, fusion: LogSpaceFusion, indices: list[int]) -> None:
        indices = sorted(set(indices))
        fusion.components = [fusion.components[i] for i in indices]
        fusion.log_weights = fusion.log_weights[indices]
        fusion.log_weights -= np.log(np.sum(np.exp(fusion.log_weights)) + 1e-12)
