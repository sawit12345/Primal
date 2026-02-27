"""Differ core knowledge: confidence-driven differentiation embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = first / (np.linalg.norm(first, axis=1, keepdims=True) + 1e-8)
    second_norm = second / (np.linalg.norm(second, axis=1, keepdims=True) + 1e-8)
    return first_norm @ second_norm.T


@dataclass
class DifferCoreKnowledge:
    """Learns embeddings that maximize confidence in differentiating classes."""

    embedding_dim: int = 32
    train_steps: int = 1800
    batch_size: int = 128
    margin: float = 0.3
    learning_rate: float = 1e-3
    seed: int = 0
    cache_path: str | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._model: Any = None
        self._image_side: int | None = None
        self._projection: np.ndarray | None = None

    def fit(self, samples: np.ndarray, labels: np.ndarray, force_retrain: bool = False) -> None:
        vectors = np.asarray(samples, dtype=np.float32).reshape(len(samples), -1)
        labels = np.asarray(labels, dtype=np.int64)

        side = int(np.sqrt(vectors.shape[1]))
        if side * side == vectors.shape[1]:
            try:
                self._fit_torch(vectors, labels, side, force_retrain=force_retrain)
                return
            except Exception:
                pass

        self._fit_linear(vectors, labels)

    def _fit_linear(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        centered = vectors - np.mean(vectors, axis=0, keepdims=True)
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[: self.embedding_dim].T

        class_means = []
        for label in sorted(set(labels.tolist())):
            class_means.append(np.mean(vectors[labels == label], axis=0))
        mean_stack = np.asarray(class_means)
        if mean_stack.shape[0] >= 2:
            class_center = mean_stack - np.mean(mean_stack, axis=0, keepdims=True)
            _, _, vh_class = np.linalg.svd(class_center, full_matrices=False)
            class_basis = vh_class[: min(self.embedding_dim, vh_class.shape[0])].T
            basis[:, : class_basis.shape[1]] = class_basis

        self._projection = basis
        self._model = None
        self._image_side = None

    def _fit_torch(self, vectors: np.ndarray, labels: np.ndarray, side: int, force_retrain: bool) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as f

        class DifferEncoder(nn.Module):
            def __init__(self, embedding_dim: int) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(32 * (side // 4) * (side // 4), 64),
                    nn.ReLU(),
                    nn.Linear(64, embedding_dim),
                )

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                embeddings = self.network(images)
                return f.normalize(embeddings, dim=1)

        torch.manual_seed(self.seed)
        model = DifferEncoder(self.embedding_dim)

        cache_path = Path(self.cache_path) if self.cache_path else None
        if cache_path is not None and cache_path.exists() and not force_retrain:
            payload = torch.load(cache_path, map_location="cpu")
            if isinstance(payload, dict) and payload.get("seed") == self.seed and payload.get("side") == side:
                model.load_state_dict(payload["state_dict"])
                model.eval()
                self._model = model
                self._image_side = side
                self._projection = None
                return

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        vectors_tensor = torch.from_numpy(vectors).reshape(-1, 1, side, side)
        labels_tensor = torch.from_numpy(labels)

        class_indices = {
            int(label): np.where(labels == label)[0]
            for label in np.unique(labels)
        }
        class_labels = np.array(sorted(class_indices.keys()), dtype=np.int64)

        for _ in range(self.train_steps):
            sampled_classes = self.rng.choice(class_labels, size=self.batch_size, replace=True)
            anchor_ids = []
            positive_ids = []
            negative_ids = []

            for class_id in sampled_classes:
                positives = class_indices[int(class_id)]
                if len(positives) == 1:
                    anchor = positives[0]
                    positive = positives[0]
                else:
                    picks = self.rng.choice(positives, size=2, replace=False)
                    anchor = int(picks[0])
                    positive = int(picks[1])

                negative_class = class_id
                while negative_class == class_id:
                    negative_class = int(self.rng.choice(class_labels))
                negative = int(self.rng.choice(class_indices[negative_class]))

                anchor_ids.append(anchor)
                positive_ids.append(positive)
                negative_ids.append(negative)

            anchor_batch = vectors_tensor[anchor_ids]
            positive_batch = vectors_tensor[positive_ids]
            negative_batch = vectors_tensor[negative_ids]

            anchor_emb = model(anchor_batch)
            positive_emb = model(positive_batch)
            negative_emb = model(negative_batch)

            triplet_loss = f.triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=self.margin, p=2)
            regularizer = 0.01 * (anchor_emb.mean(dim=0) ** 2).mean()
            loss = triplet_loss + regularizer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        self._model = model
        self._image_side = side
        self._projection = None

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": self.seed,
                    "side": side,
                    "embedding_dim": self.embedding_dim,
                },
                cache_path,
            )

    def encode(self, samples: np.ndarray, batch_size: int = 512) -> np.ndarray:
        vectors = np.asarray(samples, dtype=np.float32).reshape(len(samples), -1)

        if self._model is not None and self._image_side is not None:
            import torch

            side = self._image_side
            tensor = torch.from_numpy(vectors).reshape(-1, 1, side, side)
            embeddings: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, tensor.shape[0], batch_size):
                    batch = tensor[start : start + batch_size]
                    embeddings.append(self._model(batch).cpu().numpy())
            return np.vstack(embeddings).astype(np.float64)

        if self._projection is None:
            self._fit_linear(vectors, np.zeros(vectors.shape[0], dtype=np.int64))

        assert self._projection is not None
        embeddings = vectors @ self._projection
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return embeddings.astype(np.float64)

    def score_prototypes(self, query_embeddings: np.ndarray, prototype_embeddings: np.ndarray) -> np.ndarray:
        return _cosine_similarity(query_embeddings, prototype_embeddings)

    def confidence_from_scores(self, scores: np.ndarray) -> np.ndarray:
        sorted_scores = np.sort(scores, axis=1)
        top = sorted_scores[:, -1]
        second = sorted_scores[:, -2] if sorted_scores.shape[1] >= 2 else np.zeros_like(top)
        margin = np.clip(top - second, 0.0, 2.0)
        return margin / 2.0

    def latent_difference_confidence(self, first: np.ndarray, second: np.ndarray) -> float:
        first = np.asarray(first, dtype=np.float64).ravel()
        second = np.asarray(second, dtype=np.float64).ravel()
        delta = np.linalg.norm(first - second)
        scale = np.linalg.norm(first) + np.linalg.norm(second) + 1e-8
        return float(np.clip(delta / scale, 0.0, 1.0))
