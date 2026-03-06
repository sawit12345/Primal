"""Differ core knowledge: universal equation-driven differentiation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = first / (np.linalg.norm(first, axis=1, keepdims=True) + 1e-8)
    second_norm = second / (np.linalg.norm(second, axis=1, keepdims=True) + 1e-8)
    return first_norm @ second_norm.T


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
    return values / norms


@dataclass
class DifferCoreKnowledge:
    """Universal differentiator based on analytic equations.

    - Supervised mode (`fit` + `encode`) learns discriminative projection from
      equation-derived features using Fisher scatter equations.
    - Online universal mode (`differentiate`) keeps slot prototypes that can be
      updated on any modality without labels.
    """

    embedding_dim: int = 32
    ridge: float = 1e-3
    seed: int = 0
    online_slots: int = 12
    online_lr: float = 0.12
    online_repulsion: float = 0.04
    online_replace_threshold: float = 0.2

    feature_mean: np.ndarray | None = field(default=None, init=False)
    feature_std: np.ndarray | None = field(default=None, init=False)
    projection: np.ndarray | None = field(default=None, init=False)
    projection_bias: np.ndarray | None = field(default=None, init=False)

    online_projection: np.ndarray | None = field(default=None, init=False)
    slot_centers: np.ndarray | None = field(default=None, init=False)
    slot_counts: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _segment_pool(self, values: np.ndarray, segments: int, mode: str) -> np.ndarray:
        chunks = np.array_split(values, segments, axis=1)
        pooled = []
        for chunk in chunks:
            if chunk.shape[1] == 0:
                pooled.append(np.zeros((values.shape[0], 1), dtype=np.float64))
                continue

            if mode == "mean":
                pooled.append(np.mean(chunk, axis=1, keepdims=True))
            elif mode == "var":
                pooled.append(np.var(chunk, axis=1, keepdims=True))
            elif mode == "absmean":
                pooled.append(np.mean(np.abs(chunk), axis=1, keepdims=True))
            else:
                raise ValueError(f"Unknown pooling mode: {mode}")

        return np.concatenate(pooled, axis=1)

    def _vector_features(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        centered = vectors - np.mean(vectors, axis=1, keepdims=True)
        normalized = centered / (np.std(centered, axis=1, keepdims=True) + 1e-6)

        segments = int(np.clip(np.sqrt(vectors.shape[1]), 4, 16))
        pooled_mean = self._segment_pool(normalized, segments, "mean")
        pooled_var = self._segment_pool(normalized, segments, "var")

        first_diff = np.diff(normalized, axis=1, append=normalized[:, -1:])
        second_diff = np.diff(first_diff, axis=1, append=first_diff[:, -1:])
        pooled_diff = self._segment_pool(first_diff, segments, "absmean")
        pooled_curv = self._segment_pool(second_diff, segments, "absmean")

        fft_mag = np.abs(np.fft.rfft(normalized, axis=1))
        freq_bins = min(16, max(1, fft_mag.shape[1] - 1))
        low_freq = fft_mag[:, 1 : 1 + freq_bins]
        if low_freq.shape[1] < 16:
            low_freq = np.pad(low_freq, ((0, 0), (0, 16 - low_freq.shape[1])), mode="constant")

        stats = np.concatenate(
            [
                np.mean(np.abs(normalized), axis=1, keepdims=True),
                np.mean(normalized**2, axis=1, keepdims=True),
                np.max(normalized, axis=1, keepdims=True),
                np.min(normalized, axis=1, keepdims=True),
                np.max(normalized, axis=1, keepdims=True) - np.min(normalized, axis=1, keepdims=True),
                np.mean(normalized**3, axis=1, keepdims=True),
                np.mean(normalized**4, axis=1, keepdims=True),
                np.mean((normalized[:, :-1] * normalized[:, 1:]) < 0.0, axis=1, keepdims=True),
                np.mean(np.abs(normalized) < 0.1, axis=1, keepdims=True),
            ],
            axis=1,
        )

        return np.concatenate([pooled_mean, pooled_var, pooled_diff, pooled_curv, low_freq, stats], axis=1)

    def _image_features(self, vectors: np.ndarray, side: int) -> np.ndarray:
        images = vectors.reshape(-1, side, side)
        yy, xx = np.mgrid[0:side, 0:side]

        mass = np.sum(images, axis=(1, 2), keepdims=True) + 1e-8
        cx = np.sum(images * xx[None, :, :], axis=(1, 2), keepdims=True) / mass
        cy = np.sum(images * yy[None, :, :], axis=(1, 2), keepdims=True) / mass

        dx = xx[None, :, :] - cx
        dy = yy[None, :, :] - cy
        mu20 = np.sum(images * dx**2, axis=(1, 2), keepdims=True) / mass
        mu02 = np.sum(images * dy**2, axis=(1, 2), keepdims=True) / mass
        mu11 = np.sum(images * dx * dy, axis=(1, 2), keepdims=True) / mass

        gx = np.diff(images, axis=2, append=images[:, :, -1:])
        gy = np.diff(images, axis=1, append=images[:, -1:, :])
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_ang = np.arctan2(gy, gx)

        orientation_hist = []
        bins = 8
        for idx in range(bins):
            low = -np.pi + (2.0 * np.pi * idx / bins)
            high = -np.pi + (2.0 * np.pi * (idx + 1) / bins)
            mask = (grad_ang >= low) & (grad_ang < high)
            orientation_hist.append(np.sum(grad_mag * mask, axis=(1, 2), keepdims=True))
        orientation_hist = np.concatenate(orientation_hist, axis=1).reshape(images.shape[0], -1)
        orientation_hist /= np.sum(orientation_hist, axis=1, keepdims=True) + 1e-8

        row_proj = np.mean(images, axis=2)
        col_proj = np.mean(images, axis=1)
        row_pooled = self._segment_pool(row_proj, segments=14, mode="mean")
        col_pooled = self._segment_pool(col_proj, segments=14, mode="mean")

        radius = np.sqrt(dx**2 + dy**2)
        radius /= np.max(radius, axis=(1, 2), keepdims=True) + 1e-8
        radial = []
        for idx in range(10):
            low = idx / 10.0
            high = (idx + 1) / 10.0
            mask = (radius >= low) & (radius < high)
            radial.append(np.sum(images * mask, axis=(1, 2), keepdims=True))
        radial = np.concatenate(radial, axis=1).reshape(images.shape[0], -1)
        radial /= np.sum(radial, axis=1, keepdims=True) + 1e-8

        top, bottom = np.split(images, 2, axis=1)
        left, right = np.split(images, 2, axis=2)
        horizontal_sym = np.mean(np.abs(top - np.flip(bottom, axis=1)), axis=(1, 2), keepdims=True)
        vertical_sym = np.mean(np.abs(left - np.flip(right, axis=2)), axis=(1, 2), keepdims=True)

        binary = (images > 0.2).astype(np.float64)
        row_trans = np.mean(np.abs(np.diff(binary, axis=2)), axis=(1, 2), keepdims=True)
        col_trans = np.mean(np.abs(np.diff(binary, axis=1)), axis=(1, 2), keepdims=True)

        quadrants = [
            np.sum(images[:, : side // 2, : side // 2], axis=(1, 2), keepdims=True),
            np.sum(images[:, : side // 2, side // 2 :], axis=(1, 2), keepdims=True),
            np.sum(images[:, side // 2 :, : side // 2], axis=(1, 2), keepdims=True),
            np.sum(images[:, side // 2 :, side // 2 :], axis=(1, 2), keepdims=True),
        ]
        quadrant_mass = np.concatenate(quadrants, axis=1).reshape(images.shape[0], -1)
        quadrant_mass /= np.sum(quadrant_mass, axis=1, keepdims=True) + 1e-8

        moments = np.concatenate([mass / (side * side), cx / side, cy / side, mu20, mu02, mu11], axis=1)
        moments = moments.reshape(images.shape[0], -1)
        sym_trans = np.concatenate([horizontal_sym, vertical_sym, row_trans, col_trans], axis=1)
        sym_trans = sym_trans.reshape(images.shape[0], -1)

        return np.concatenate([row_pooled, col_pooled, orientation_hist, radial, quadrant_mass, moments, sym_trans], axis=1)

    def _equation_features(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        universal = self._vector_features(vectors)

        feature_dim = vectors.shape[1]
        side = int(np.sqrt(feature_dim))
        if side * side != feature_dim:
            return universal

        image_specific = self._image_features(vectors, side)
        return np.concatenate([universal, image_specific], axis=1)

    def fit(self, samples: np.ndarray, labels: np.ndarray, force_retrain: bool = False) -> None:
        del force_retrain
        vectors = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
        labels = np.asarray(labels, dtype=np.int64)

        features = self._equation_features(vectors)
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0) + 1e-6
        normalized = (features - feature_mean[None, :]) / feature_std[None, :]

        classes = np.unique(labels)
        global_mean = np.mean(normalized, axis=0)
        dim = normalized.shape[1]

        scatter_within = self.ridge * np.eye(dim, dtype=np.float64)
        scatter_between = np.zeros((dim, dim), dtype=np.float64)

        for class_label in classes:
            class_samples = normalized[labels == class_label]
            if class_samples.shape[0] <= 1:
                continue

            class_mean = np.mean(class_samples, axis=0)
            centered = class_samples - class_mean[None, :]
            scatter_within += centered.T @ centered

            mean_delta = (class_mean - global_mean)[:, None]
            scatter_between += class_samples.shape[0] * (mean_delta @ mean_delta.T)

        eigenvals, eigenvecs = eigh(scatter_between, scatter_within)
        order = np.argsort(eigenvals)[::-1]
        chosen = order[: min(self.embedding_dim, eigenvecs.shape[1])]
        projection = eigenvecs[:, chosen]

        projected = normalized @ projection
        projected_mean = np.mean(projected, axis=0)

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.projection = projection
        self.projection_bias = projected_mean

        self.online_projection = None
        self.slot_centers = None
        self.slot_counts = None

    def encode(self, samples: np.ndarray, batch_size: int = 512) -> np.ndarray:
        del batch_size
        vectors = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
        features = self._equation_features(vectors)

        if self.feature_mean is None or self.feature_std is None:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-6

        feature_mean = np.asarray(self.feature_mean, dtype=np.float64)
        feature_std = np.asarray(self.feature_std, dtype=np.float64)
        normalized = (features - feature_mean[None, :]) / feature_std[None, :]

        if self.projection is None:
            _, _, vh = np.linalg.svd(normalized, full_matrices=False)
            projection = vh[: min(self.embedding_dim, vh.shape[0])].T
            self.projection = projection
            self.projection_bias = np.zeros(projection.shape[1], dtype=np.float64)

        assert self.projection is not None
        if self.projection_bias is None:
            self.projection_bias = np.zeros(self.projection.shape[1], dtype=np.float64)

        embedding = normalized @ self.projection
        embedding = embedding - self.projection_bias[None, :]
        embedding = _normalize_rows(embedding)
        return np.asarray(embedding, dtype=np.float64)

    def _ensure_online_state(self, feature_dim: int) -> None:
        if self.online_projection is None or self.online_projection.shape[0] != feature_dim:
            raw = self._rng.normal(0.0, 1.0 / np.sqrt(max(1, feature_dim)), size=(feature_dim, self.embedding_dim))
            q, _ = np.linalg.qr(raw)
            if q.shape[1] < self.embedding_dim:
                padding = self._rng.normal(0.0, 1.0 / np.sqrt(max(1, feature_dim)), size=(feature_dim, self.embedding_dim - q.shape[1]))
                projection = np.concatenate([q, padding], axis=1)
            else:
                projection = q[:, : self.embedding_dim]

            self.online_projection = projection
            centers = self._rng.normal(0.0, 1.0, size=(self.online_slots, self.embedding_dim))
            self.slot_centers = _normalize_rows(centers)
            self.slot_counts = np.full(self.online_slots, 1e-3, dtype=np.float64)

    def differentiate(self, sample: np.ndarray, learn: bool = True) -> tuple[np.ndarray, float, np.ndarray]:
        vector = np.asarray(sample, dtype=np.float64).reshape(1, -1)
        features = self._equation_features(vector)

        if self.feature_mean is not None and self.feature_std is not None and self.feature_mean.shape[0] == features.shape[1]:
            standardized = (features - self.feature_mean[None, :]) / (self.feature_std[None, :] + 1e-6)
        else:
            standardized = (features - np.mean(features, axis=1, keepdims=True)) / (
                np.std(features, axis=1, keepdims=True) + 1e-6
            )

        self._ensure_online_state(standardized.shape[1])
        assert self.online_projection is not None
        assert self.slot_centers is not None
        assert self.slot_counts is not None

        embedding = standardized @ self.online_projection
        embedding = _normalize_rows(embedding)
        embed_vec = embedding[0]

        similarities = self.slot_centers @ embed_vec
        ranking = np.argsort(similarities)
        top_idx = int(ranking[-1])
        second_idx = int(ranking[-2]) if similarities.shape[0] >= 2 else top_idx

        margin = float(np.clip(similarities[top_idx] - similarities[second_idx], 0.0, 2.0))
        confidence = margin / 2.0

        if learn:
            if similarities[top_idx] < self.online_replace_threshold:
                weakest = int(np.argmin(self.slot_counts))
                self.slot_centers[weakest] = embed_vec
                self.slot_counts[weakest] = 1.0
            else:
                count = self.slot_counts[top_idx] + 1.0
                eta = self.online_lr / np.sqrt(count)
                updated = (1.0 - eta) * self.slot_centers[top_idx] + eta * embed_vec
                self.slot_centers[top_idx] = updated / (np.linalg.norm(updated) + 1e-8)
                self.slot_counts[top_idx] = count

                if second_idx != top_idx:
                    repelled = self.slot_centers[second_idx] - self.online_repulsion * eta * embed_vec
                    self.slot_centers[second_idx] = repelled / (np.linalg.norm(repelled) + 1e-8)

        return embed_vec, confidence, similarities

    def score_prototypes(self, query_embeddings: np.ndarray, prototype_embeddings: np.ndarray) -> np.ndarray:
        return _cosine_similarity(query_embeddings, prototype_embeddings)

    def confidence_from_scores(self, scores: np.ndarray) -> np.ndarray:
        sorted_scores = np.sort(scores, axis=1)
        top = sorted_scores[:, -1]
        second = sorted_scores[:, -2] if sorted_scores.shape[1] >= 2 else np.zeros_like(top)
        margin = np.clip(top - second, 0.0, 2.0)
        return margin / 2.0

    def latent_difference_confidence(self, first: np.ndarray, second: np.ndarray) -> float:
        first_emb, first_conf, _ = self.differentiate(first, learn=False)
        second_emb, second_conf, _ = self.differentiate(second, learn=False)

        cosine = float(np.clip(np.dot(first_emb, second_emb), -1.0, 1.0))
        shape_difference = (1.0 - cosine) / 2.0
        confidence_gap = abs(first_conf - second_conf)
        return float(np.clip(0.8 * shape_difference + 0.2 * confidence_gap, 0.0, 1.0))
