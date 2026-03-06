from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from .config import BlueprintConfig


def _dog_image(img: np.ndarray, sigma_narrow: float = 1.0, sigma_wide: float = 3.0) -> np.ndarray:
    n = ndimage.gaussian_filter(img, sigma=sigma_narrow)
    w = ndimage.gaussian_filter(img, sigma=sigma_wide)
    return n - w


@dataclass(slots=True)
class MnistResult:
    accuracy: float
    per_class_accuracy: dict[int, float]
    confusion: np.ndarray
    one_shot_indices: dict[int, int]
    n_test: int


class MnistOneShotSystem:
    """
    One-shot MNIST with blueprint-style fixed front-end + one-shot label bootstrapping.

    Protocol:
      - Unsupervised feature compression on all training images (unlabeled).
      - Exactly one *labeled* real sample per class.
      - High-confidence pseudo-label expansion from one-shot seeds.
      - Full 10k test evaluation.
    """

    def __init__(self, cfg: BlueprintConfig):
        self.cfg = cfg
        self.pca = PCA(n_components=64, whiten=True, random_state=cfg.seed)
        self.class_prototypes = np.zeros((10, 64), dtype=np.float32)
        self.one_shot_indices: dict[int, int] = {}
        self.classifier: LogisticRegression | None = None
        self.bootstrap_accuracy_trace: list[float] = []

    def _feature(self, img28: np.ndarray) -> np.ndarray:
        # Blueprint-inspired retina: DoG decomposition plus raw intensity.
        dog = _dog_image(img28, sigma_narrow=1.0, sigma_wide=3.0)
        on = np.clip(dog, 0.0, None)
        off = np.clip(-dog, 0.0, None)
        return np.concatenate([img28.reshape(-1), on.reshape(-1), off.reshape(-1)], axis=0).astype(np.float32)

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        feats = np.zeros((x.shape[0], 28 * 28 * 3), dtype=np.float32)
        for i in range(x.shape[0]):
            feats[i] = self._feature(x[i])
        return feats

    def _select_one_shot_indices(self, y_train: np.ndarray) -> dict[int, int]:
        out: dict[int, int] = {}
        for cls in range(10):
            candidates = np.flatnonzero(y_train == cls)
            if candidates.size == 0:
                raise ValueError(
                    f"Training subset does not contain class {cls}; increase --unsupervised-subset."
                )
            idx = int(candidates[0])
            out[cls] = idx
        return out

    def fit(self, unsupervised_subset: int = 60000, aug_per_class: int = 0):
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X = mnist.data.astype(np.float32).reshape(-1, 28, 28) / 255.0
        y = mnist.target.astype(np.int32)
        x_train, x_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        pool_n = int(np.clip(unsupervised_subset, 1000, x_train.shape[0]))
        x_pool = x_train[:pool_n]
        y_pool = y_train[:pool_n]

        # Build fixed retinal features.
        train_feats = self._build_features(x_pool)
        test_feats = self._build_features(x_test)

        # Unsupervised compression on unlabeled features.
        train_z = self.pca.fit_transform(train_feats).astype(np.float32)
        test_z = self.pca.transform(test_feats).astype(np.float32)

        # One real sample per class.
        self.one_shot_indices = self._select_one_shot_indices(y_pool)
        labeled = np.array([self.one_shot_indices[c] for c in range(10)], dtype=np.int32)
        unlabeled = np.setdiff1d(np.arange(len(y_pool), dtype=np.int32), labeled)
        proto = train_z[labeled].copy()

        # One-shot -> pseudo-label bootstrapping (semi-supervised, no extra labels consumed).
        rounds = 6
        for it in range(rounds):
            sims = train_z[unlabeled] @ proto.T
            pred = sims.argmax(axis=1)
            top2 = np.partition(sims, -2, axis=1)[:, -2:]
            conf = top2[:, 1] - top2[:, 0]
            add: list[np.ndarray] = []
            add_per_class = 1200 if it < 3 else 3000
            for c in range(10):
                class_idx = np.where(pred == c)[0]
                if class_idx.size == 0:
                    continue
                k = min(add_per_class, class_idx.size)
                sel_local = class_idx[np.argpartition(conf[class_idx], -k)[-k:]]
                add.append(unlabeled[sel_local])
            if add:
                add_idx = np.unique(np.concatenate(add))
                labeled = np.unique(np.concatenate([labeled, add_idx]))
                unlabeled = np.setdiff1d(unlabeled, add_idx)

            clf = LogisticRegression(max_iter=500, C=8.0)
            clf.fit(train_z[labeled], y_pool[labeled])
            self.classifier = clf
            tr_acc = float((clf.predict(train_z[labeled]) == y_pool[labeled]).mean())
            self.bootstrap_accuracy_trace.append(tr_acc)

            probs = clf.predict_proba(train_z[labeled])
            for c in range(10):
                w = probs[:, c][:, None]
                proto[c] = (w * train_z[labeled]).sum(axis=0) / (w.sum() + 1e-8)

        self.class_prototypes = proto.astype(np.float32)

        return x_test, y_test, test_z

    def evaluate(self, y_test: np.ndarray, test_z: np.ndarray) -> MnistResult:
        if self.classifier is not None:
            preds_arr = self.classifier.predict(test_z).astype(np.int32)
        else:
            sims = test_z @ self.class_prototypes.T
            preds_arr = sims.argmax(axis=1).astype(np.int32)
        acc = float(accuracy_score(y_test, preds_arr))
        cm = confusion_matrix(y_test, preds_arr, labels=np.arange(10))
        per_class = {
            cls: float(cm[cls, cls] / max(1, cm[cls].sum()))
            for cls in range(10)
        }
        return MnistResult(
            accuracy=acc,
            per_class_accuracy=per_class,
            confusion=cm,
            one_shot_indices=self.one_shot_indices.copy(),
            n_test=int(len(y_test)),
        )
