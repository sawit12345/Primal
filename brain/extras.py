"""Additional lightweight brain-inspired modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThalamicPrecisionGate:
    """Routes latent activity with precision-dependent gain."""

    baseline: float = 1.0

    def gate(self, latent: np.ndarray, precision: np.ndarray | float) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float64)
        precision_array = np.asarray(precision, dtype=np.float64)
        return latent * (self.baseline + precision_array)


@dataclass
class NeuromodulatorySwitch:
    """Produces smooth exploration modulation from prediction stress."""

    gain: float = 0.6

    def modulate(self, prediction_error: float, free_energy: float) -> float:
        signal = float(prediction_error + free_energy)
        return float(1.0 + self.gain * np.tanh(signal))
