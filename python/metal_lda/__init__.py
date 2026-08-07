"""metal_lda — Python wrapper for the Metal-accelerated LDA operator.

Provides a scikit-learn-compatible API mirroring ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis``
(dimensionality reduction + classification via nearest projected class center).

Usage::

    from metal_lda import MetalLDA
    import numpy as np

    X = np.random.randn(300, 8).astype(np.float32)
    y = np.array([0] * 150 + [1] * 150, dtype=np.float32)

    lda = MetalLDA(n_components=1)
    lda.fit(X, y)              # data: numpy array or list-of-lists
    X_red = lda.transform(X)   # (n, 1)
    preds = lda.predict(X)     # class indices 0..C-1
    acc = lda.score(X, y)      # fraction correct

    scalings = lda.scalings_   # (k, d) discriminant axes
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalLDA as _MetalLDA
from metal_kmeans._native import metal_lda_fit_bytes as _metal_lda_fit

__all__ = ["MetalLDA", "metal_lda"]


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()


class MetalLDA:
    """Linear discriminant analysis with Metal GPU acceleration.

    Parameters
    ----------
    n_components : int
        Number of discriminant axes to keep. Clamped to
        ``min(n_classes - 1, n_features)``.
    """

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self._model: _MetalLDA | None = None

    def fit(
        self,
        X: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> "MetalLDA":
        """Fit LDA to data *X* with class labels *y*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or flat list.
        y : ndarray/list of class labels, shape (n_samples,).
        n, d : optional explicit shape; inferred from ``np.asarray(X).shape``.
        """
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32).ravel()
        if n is None or d is None:
            n, d = Xa.shape
        if ya.size != n:
            raise ValueError(f"labels size {ya.size} != n_samples {n}")

        flat = np.ascontiguousarray(Xa, dtype=np.float32).tobytes()
        labels = np.ascontiguousarray(ya, dtype=np.float32).tobytes()

        self._model = _MetalLDA(self.n_components)
        self._model.fit_bytes(flat, labels, n, d)

        k = self._model.n_components
        scal = self._model.scalings
        self.scalings_ = np.array(scal, dtype=np.float32).reshape(k, d)
        self.coef_ = self.scalings_.copy()
        self.eigenvalues_ = np.array(self._model.eigenvalues, dtype=np.float32)
        self.classes_ = np.array(self._model.classes, dtype=np.float32)
        self.n_features_in_ = d
        self.n_components = k
        return self

    def transform(
        self,
        X: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> np.ndarray:
        """Project *X* into the discriminant subspace: (n, k)."""
        X = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = X.shape
        if self._model is None:
            raise RuntimeError("MetalLDA is not fitted")
        raw = self._model.transform_bytes(
            np.ascontiguousarray(X, dtype=np.float32).tobytes(), n, d
        )
        return np.array(raw, dtype=np.float32).reshape(n, self.n_components)

    def fit_transform(self, X, y) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: np.ndarray | list[float]) -> np.ndarray:
        """Return class indices (0..C-1) for the rows of *X*."""
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        if self._model is None:
            raise RuntimeError("MetalLDA is not fitted")
        raw = self._model.predict_bytes(
            np.ascontiguousarray(X, dtype=np.float32).tobytes(), n, d
        )
        return np.asarray(raw, dtype=np.int64)

    def score(self, X, y) -> float:
        """Accuracy of nearest-project-class-center predictions vs *y*."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n, d = X.shape
        if self._model is None:
            raise RuntimeError("MetalLDA is not fitted")
        return self._model.score_bytes(
            np.ascontiguousarray(X, dtype=np.float32).tobytes(),
            np.ascontiguousarray(y, dtype=np.float32).tobytes(),
            n,
            d,
        )


def metal_lda(
    data: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    n: int,
    d: int,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit LDA on the GPU and return (scalings, eigenvalues, classes)."""
    flat = _as_bytes_f32(data)
    flat_y = _as_bytes_f32(labels)
    scal, ev, classes = _metal_lda_fit(flat, flat_y, n, d, n_components)
    return (
        np.array(scal, dtype=np.float32).reshape(n_components, d),
        np.array(ev, dtype=np.float32),
        np.array(classes, dtype=np.float32),
    )