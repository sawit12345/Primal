from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def cosine_sim_matrix(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    qn = query / (np.linalg.norm(query) + 1e-8)
    kn = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
    return kn @ qn


class DentateGyrus:
    """
    Fixed random expansion projection + top-k threshold (2% default).
    """

    def __init__(
        self,
        input_dim: int = 384,
        output_dim: int = 3840,
        sparsity: float = 0.02,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(output_dim, input_dim)).astype(
            np.float32
        )

    def project(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.input_dim:
            raise ValueError(f"DentateGyrus expected {self.input_dim} dims, got {x.shape[0]}")
        z = self.W @ x.astype(np.float32)
        k = max(1, int(self.output_dim * self.sparsity))
        idx = np.argpartition(z, -k)[-k:]
        out = np.zeros_like(z, dtype=np.float32)
        out[idx] = 1.0
        return out


@dataclass(slots=True)
class RetrievalResult:
    vector: np.ndarray
    index: int
    confidence: float
    top_indices: np.ndarray
    top_scores: np.ndarray


class HippocampalBuffer:
    """
    One-shot episodic storage with masked pattern completion and replay clustering.
    """

    def __init__(self, dim: int = 1152, capacity: int = 10000):
        self.dim = dim
        self.capacity = capacity
        self.data = np.zeros((capacity, dim), dtype=np.float32)
        self.social_flags = np.zeros(capacity, dtype=bool)
        self.novelty = np.zeros(capacity, dtype=np.float32)
        self.recency = np.zeros(capacity, dtype=np.float32)
        self.retrieval_count = np.zeros(capacity, dtype=np.int32)
        self.ptr = 0
        self.size = 0
        self.time = 0
        self.cluster_ids = np.full(capacity, -1, dtype=np.int32)

    def write(self, x: np.ndarray, novelty: float, social: bool = False) -> int:
        if x.shape[0] != self.dim:
            raise ValueError(f"Hippocampal tuple must have dim={self.dim}, got {x.shape[0]}")
        idx = self.ptr
        self.data[idx] = x.astype(np.float32)
        self.social_flags[idx] = social
        self.novelty[idx] = float(novelty)
        self.recency[idx] = float(self.time)
        self.retrieval_count[idx] = 0
        self.cluster_ids[idx] = -1

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        self.time += 1
        return idx

    def retrieve(self, query: np.ndarray, mask: np.ndarray | None = None, top_k: int = 5) -> RetrievalResult:
        if self.size == 0:
            return RetrievalResult(
                vector=np.zeros(self.dim, dtype=np.float32),
                index=-1,
                confidence=0.0,
                top_indices=np.array([], dtype=np.int32),
                top_scores=np.array([], dtype=np.float32),
            )
        q = query.astype(np.float32)
        keys = self.data[: self.size]
        if mask is not None:
            q = q * mask
            keys = keys * mask[None, :]

        sims = cosine_sim_matrix(q, keys)
        k = min(top_k, self.size)
        top = np.argpartition(sims, -k)[-k:]
        top = top[np.argsort(sims[top])[::-1]]
        best = int(top[0])
        confidence = float(max(0.0, sims[best]))
        self.retrieval_count[top] += 1
        return RetrievalResult(
            vector=self.data[best].copy(),
            index=best,
            confidence=confidence,
            top_indices=top.astype(np.int32),
            top_scores=sims[top].astype(np.float32),
        )

    def retention_scores(self) -> np.ndarray:
        if self.size == 0:
            return np.array([], dtype=np.float32)
        rec = self.recency[: self.size]
        freq = self.retrieval_count[: self.size].astype(np.float32)
        rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-8)
        freq_norm = (freq - freq.min()) / (freq.max() - freq.min() + 1e-8)
        return 0.5 * rec_norm + 0.5 * freq_norm

    def prune_low_retention(self, frac: float = 0.1):
        if self.size < 10:
            return
        scores = self.retention_scores()
        n_drop = max(1, int(self.size * frac))
        drop_idx = np.argpartition(scores, n_drop)[:n_drop]
        keep_mask = np.ones(self.size, dtype=bool)
        keep_mask[drop_idx] = False
        kept = self.data[: self.size][keep_mask]
        kept_social = self.social_flags[: self.size][keep_mask]
        kept_novelty = self.novelty[: self.size][keep_mask]
        kept_rec = self.recency[: self.size][keep_mask]
        kept_cnt = self.retrieval_count[: self.size][keep_mask]
        kept_cluster = self.cluster_ids[: self.size][keep_mask]

        new_size = kept.shape[0]
        self.data[:new_size] = kept
        self.social_flags[:new_size] = kept_social
        self.novelty[:new_size] = kept_novelty
        self.recency[:new_size] = kept_rec
        self.retrieval_count[:new_size] = kept_cnt
        self.cluster_ids[:new_size] = kept_cluster
        self.size = new_size
        self.ptr = new_size % self.capacity

    def build_centroids(self, k: int = 16, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32), np.zeros(0, dtype=np.int32)
        k_eff = min(k, self.size)
        km = MiniBatchKMeans(n_clusters=k_eff, random_state=random_state, n_init="auto", batch_size=512)
        labels = km.fit_predict(self.data[: self.size])
        centroids = km.cluster_centers_.astype(np.float32)
        self.cluster_ids[: self.size] = labels
        return centroids, labels.astype(np.int32)
