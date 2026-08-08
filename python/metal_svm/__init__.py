"""metal_svm — GPU-accelerated Support Vector Classification (SVC) via Apple Metal.

Provides a sklearn-style class (``MetalSVC``) and a functional fit API
(``metal_svc``), mirroring ``sklearn.svm.SVC``. The two heavy linear-algebra
stages run on the GPU:

1. the full ``n×n`` kernel (Gram) matrix ``K[i][j] = κ(x_i, x_j)`` is built in
   a single launch (``svm_kernel`` in ``shaders/svm.metal``) and, since it
   depends only on the data (not the labels), is reused verbatim by every
   one-vs-rest binary sub-problem;
2. the decision matrix is computed in a single launch (``svm_predict``)
   against the pooled support vectors.

The per-iteration scalar dual updates run on the host as a simplified Platt
**Sequential Minimal Optimization (SMO)** solver.

Usage::

    from metal_svm import MetalSVC, metal_svc
    import numpy as np

    X = np.array([
        [0.0, 0.0], [1.0, 1.0],   # class 0
        [0.0, 1.0], [1.0, 0.0],   # class 1 (XOR, not linearly separable)
    ], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.float32)

    # sklearn-style API
    clf = MetalSVC(kernel="rbf", gamma=0.5, c=100.0)   # gamma = kernel width
    clf.fit(X, y)
    preds = clf.predict(X)          # (n,)
    dec = clf.decision_function(X)  # (n, n_classes) raw scores

    # Functional API — returns
    # (classes, intercept_, dual_coef_, support_vectors_, support_count, gamma_, n_iter)
    classes, intercept, dual, sv, ns, g, iters = metal_svc(
        X, y, *X.shape, kernel="rbf", gamma=0.5, c=100.0
    )

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of shape
``(n, d)`` in row-major order; ``y`` must be ``(n,)`` class labels (any
``float`` values; labels that differ by less than 0.5 count as the same class).

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalSVC as _MetalSVC
from metal_kmeans._native import metal_svc_fit_bytes as _metal_svc_fit

__all__ = ["MetalSVC", "metal_svc"]


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    """Return raw little-endian float32 bytes of *data* (C-speed memcpy).

    Consumed by the ``_bytes`` bindings — avoiding the O(n) ``tolist()``
    roundtrip that the ``Vec<f32>`` methods pay (multi-second on 1M+ samples).
    """
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()


class MetalSVC:
    """sklearn-style Support Vector Classifier using GPU-accelerated Metal.

    Parameters
    ----------
    kernel : str, default="rbf"
        Kernel type: ``"linear"``, ``"poly"``, ``"rbf"`` or ``"sigmoid"``.
    gamma : float, default=0.0
        Manual kernel width. ``<= 0`` selects the automatic default
        ``1 / n_features`` (the ``1/n_features`` fallback of ``gamma="scale"``).
    degree : float, default=3.0
        Polynomial degree (only used when ``kernel="poly"``).
    coef0 : float, default=0.0
        Independent term in poly / sigmoid kernels.
    c : float, default=1.0
        Regularization parameter (penalty on misclassification).
    tolerance : float, default=1e-3
        SMO convergence tolerance.
    max_iter : int, default=200
        Maximum number of SMO passes.
    seed : int, default=42
        Seed for the (reproducible) SMO index sampling.

    Attributes
    ----------
    classes_ : ndarray (n_classes,), float32
        Unique class labels in the order they appear during ``fit``.
    n_support_ : ndarray (n_classes,), int
        Number of support vectors per one-vs-rest classifier.
    support_vectors_ : ndarray (n_support, n_features), float32
        Pooled support vectors across all classifiers.
    dual_coef_ : ndarray (n_support,), float32
        Pooled ``α·y`` dual coefficients aligned to ``support_vectors_``.
    intercept_ : ndarray (n_classes,), float32
        Per-classifier intercept ``b``.
    n_iter_ : ndarray (n_classes,), int
        SMO passes run per classifier.
    gamma_ : float
        Resolved kernel width used by ``fit``.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float = 0.0,
        degree: float = 3.0,
        coef0: float = 0.0,
        c: float = 1.0,
        tolerance: float = 1e-3,
        max_iter: int = 200,
        seed: int = 42,
    ) -> None:
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.c = c
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.seed = seed
        self._model: _MetalSVC | None = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        X: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> "MetalSVC":
        """Fit the SVC with the GPU Gram matrix + host SMO.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or flat list (float32).
        y : ndarray of shape (n_samples,) class labels.
        n, d : optional explicit shape; inferred from ``np.asarray(X).shape``.
        """
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        flat = np.ascontiguousarray(Xa, dtype=np.float32).tobytes()
        y_arr = np.ascontiguousarray(y, dtype=np.float32).tobytes()

        model = _MetalSVC(
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
            self.c,
            self.tolerance,
            self.max_iter,
            self.seed,
        )
        model.fit_bytes(flat, y_arr, n, d)
        self._model = model
        self.n_features_in_ = d

        self.classes_ = np.array(model.classes, dtype=np.float32)
        self.intercept_ = np.array(model.intercept_, dtype=np.float32)
        self.n_support_ = np.array(model.n_support, dtype=np.intp)
        ns = int(sum(model.n_support))
        self.support_vectors_ = np.array(
            model.support_vectors_, dtype=np.float32
        ).reshape(ns, d)
        self.dual_coef_ = np.array(model.dual_coef_, dtype=np.float32)
        self.n_iter_ = np.array(model.n_iter, dtype=np.intp)
        self.gamma_ = model.gamma_
        return self

    def predict(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> np.ndarray:
        """Predict hard class labels (argmax / sign of the decision matrix).

        Returns labels of shape ``(n,)`` with dtype intp.
        """
        if self._model is None:
            raise RuntimeError("MetalSVC is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        raw = self._model.predict_bytes(
            np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), n, d
        )
        return np.array(raw, dtype=np.intp)

    def decision_function(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> np.ndarray:
        """Raw decision scores of shape ``(n, n_classes)`` float32.

        For the two-class case the single column is returned (signed score).
        """
        if self._model is None:
            raise RuntimeError("MetalSVC is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        raw = self._model.decision_function_bytes(
            np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), n, d
        )
        dec = np.array(raw, dtype=np.float32).reshape(n, self.n_support_.size)
        if dec.shape[1] == 1:
            dec = dec[:, 0]
        return dec

    def score(
        self,
        X: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> float:
        """Return mean accuracy of ``predict(X)`` against *y*."""
        if self._model is None:
            raise RuntimeError("MetalSVC is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        y_arr = np.ascontiguousarray(y, dtype=np.float32).tobytes()
        return float(
            self._model.score_bytes(
                np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), y_arr, n, d
            )
        )


def metal_svc(
    data: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n: int,
    d: int,
    kernel: str = "linear",
    gamma: float = 20.0,
    degree: float = 3.0,
    coef0: float = 0.0,
    c: float = 1.0,
    tolerance: float = 1e-4,
    max_iter: int = 25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray]:
    """Fit an SVC on the GPU and return its model parameters.

    .. note::
       The functional API does not retain the fitted model for later
       prediction; use the ``MetalSVC`` class when you need
       ``predict`` / ``decision_function`` / ``score``.

    Returns
    -------
    classes : ndarray (n_classes,) float32
    intercept_ : ndarray (n_classes,) float32
    dual_coef_ : ndarray (n_support,) float32
    support_vectors_ : ndarray (n_support, d) float32
    support_count : int
    gamma_ : float
    n_iter_ : ndarray (n_classes,) int
    """
    flat = _as_bytes_f32(data)
    y_arr = _as_bytes_f32(y)
    classes, intercept, dual, sv, count, g, iters = _metal_svc_fit(
        flat,
        y_arr,
        n,
        d,
        kernel,
        gamma,
        degree,
        coef0,
        c,
        tolerance,
        max_iter,
        seed,
    )
    return (
        np.array(classes, dtype=np.float32),
        np.array(intercept, dtype=np.float32),
        np.array(dual, dtype=np.float32),
        np.array(sv, dtype=np.float32).reshape(count, d),
        count,
        g,
        np.array(iters, dtype=np.intp),
    )