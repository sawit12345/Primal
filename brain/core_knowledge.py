"""Core-knowledge and transfer-learning substrate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.temporal_decay import MarkovTemporalDecay


@dataclass
class _CoreChannel:
    name: str
    projection: np.ndarray
    prototype: np.ndarray
    scale: float = 1.0


@dataclass
class CoreKnowledgeTransfer:
    """Learns reusable latent priors for object, space, number, agent, and physics channels."""

    input_dim: int
    channel_dim: int = 16
    seed: int = 0
    decay: MarkovTemporalDecay = field(default_factory=MarkovTemporalDecay)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        names = [
            "objects",
            "space_geometry",
            "number",
            "agents",
            "physics",
            "causality",
            "affordance",
        ]
        self.channels: list[_CoreChannel] = []
        for name in names:
            projection = rng.normal(0.0, 0.2, size=(self.channel_dim, self.input_dim))
            prototype = np.zeros(self.channel_dim, dtype=np.float64)
            self.channels.append(_CoreChannel(name=name, projection=projection, prototype=prototype))

    def encode_channels(self, features: np.ndarray) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float64).ravel()
        encodings: dict[str, np.ndarray] = {}
        for channel in self.channels:
            projected = channel.projection @ features
            encodings[channel.name] = np.tanh(channel.scale * projected)
        return encodings

    def update(self, features: np.ndarray) -> None:
        encodings = self.encode_channels(features)
        for channel in self.channels:
            current = encodings[channel.name]
            channel.prototype = self.decay.blend(channel.prototype, current)
            prototype_norm = np.linalg.norm(channel.prototype) + 1e-8
            channel.scale = float(np.clip(prototype_norm / np.sqrt(channel.prototype.size), 0.5, 2.5))

    def transfer_embedding(self, features: np.ndarray) -> np.ndarray:
        encodings = self.encode_channels(features)
        transfer_chunks: list[np.ndarray] = []
        for channel in self.channels:
            channel_state = encodings[channel.name]
            similarity = np.dot(channel_state, channel.prototype) / (
                np.linalg.norm(channel_state) * np.linalg.norm(channel.prototype) + 1e-8
            )
            transfer_chunks.append(channel_state)
            transfer_chunks.append(np.array([similarity], dtype=np.float64))
        return np.concatenate(transfer_chunks, axis=0)
