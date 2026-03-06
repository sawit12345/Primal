from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VisionConfig:
    input_size: int = 84
    dog_sigma_narrow: float = 1.0
    dog_sigma_wide: float = 3.0
    gabor_orientations: int = 8
    gabor_scales: tuple[float, ...] = (2.0, 4.0, 8.0)
    gabor_kernel_size: int = 15
    v1_threshold_frac: float = 0.2
    v5_grid: int = 16


@dataclass(slots=True)
class RGMLevelConfig:
    n_states: int
    max_states: int
    block_nonzero: int
    n_paths: int
    growth_threshold: float


@dataclass(slots=True)
class MemoryConfig:
    dg_input_dim: int = 384
    dg_output_dim: int = 3840
    dg_sparsity: float = 0.02
    hippocampal_dim: int = 1152
    hippocampal_capacity: int = 10000
    write_novelty_threshold: float = 0.6


@dataclass(slots=True)
class CortexConfig:
    input_dim: int = 640
    layer_dims: tuple[int, ...] = (512, 256, 256, 256, 256)
    language_dim: int = 256
    learning_rate: float = 0.005
    error_gate_threshold: float = 0.05


@dataclass(slots=True)
class ActionConfig:
    n_actions: int = 64
    td_lambda: float = 0.9
    gamma: float = 0.95


@dataclass(slots=True)
class BlueprintConfig:
    seed: int = 7
    vision: VisionConfig = field(default_factory=VisionConfig)
    rgm_levels: tuple[RGMLevelConfig, ...] = field(
        default_factory=lambda: (
            RGMLevelConfig(
                n_states=64,
                max_states=160,
                block_nonzero=8,
                n_paths=4,
                growth_threshold=0.45,
            ),
            RGMLevelConfig(
                n_states=128,
                max_states=320,
                block_nonzero=8,
                n_paths=4,
                growth_threshold=0.40,
            ),
            RGMLevelConfig(
                n_states=256,
                max_states=512,
                block_nonzero=8,
                n_paths=4,
                growth_threshold=0.35,
            ),
        )
    )
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cortex: CortexConfig = field(default_factory=CortexConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
