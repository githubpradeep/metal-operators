"""metal_svr — GPU-accelerated Support Vector Regression (SVR) via Apple Metal.

Provides a sklearn-style class (``MetalSVR``) and a functional fit API
(``metal_svr``), mirroring ``sklearn.svm.SVR``. The two heavy linear-algebra
stages run on the GPU:

1. the full ``n×n`` kernel (Gram) matrix ``K[i][j] = κ(x_i, x_j)`` is built in
   a single launch (``svm_kernel`` in ``shaders/svm.metal``) — the same shader
   reused by :mod:`metal_svm`;
2. the regression outputs are computed in a single launch (``svm_predict``)
   against the pooled support vectors.

The per-iteration scalar dual updates run on the host as an ε-insensitive
**Sequential Optimization (SMO)** solver on ``β_i = α_i⁺ - α_i⁻ ∈ [-C, C]``
with ``Σβ = 0`` and the ε-tube penalty.

Usage::

    from metal_svr import MetalSVR, metal_svr
    import numpy as np

    # 1-D sine — RBF kernel fits the non-linear target.
    t = np.linspace(0, 6.28, 120).astype(np.float32)
    X = t.reshape(-1, 1)
    y = np.sin(t).astype(np.float32)

    clf = MetalSVR(kernel="rbf", gamma=0.6, c=5.0, eps=0.05)
    clf.fit(X, y)
    r2 = clf.score(X, y)
    preds = clf.predict(X)          # (n,)

    # Functional API — returns
    # (support_vectors_, dual_coef_/β, intercept_, support_count, gamma_, n_iter)
    sv, dual, b, ns, g, iters = metal_svr(X, y, *X.shape,
                                          kernel="rbf", gamma=0.6, c=5.0)

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of shape
``(n, d)`` in row-major order; ``y`` must be ``(n,)`` float regression targets.

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalSVR as _MetalSVR
from metal_kmeans._native import metal_svr_fit_bytes as _metal_svr_fit

__all__ = ["MetalSVR", "metal_svr"]


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    """Return raw little-endian float32 bytes of *data* (C-speed memcpy).

    Consumed by the ``_bytes`` bindings — avoiding the O(n) ``tolist()``
    roundtrip that the ``Vec<f32>`` methods pay (multi-second on 1M+ samples).
    """
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()


class MetalSVR:
    """sklearn-style Support Vector Regressor using GPU-accelerated Metal.

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
        Regularization parameter (penalty on ε-tube violations).
    eps : float, default=0.1
        ε-insensitive tube width: residuals within ``eps`` cost nothing.
    tolerance : float, default=1e-3
        SMO convergence tolerance.
    max_iter : int, default=200
        Maximum number of SMO passes.
    seed : int, default=42
        Seed for the (reproducible) SMO index sampling.

    Attributes
    ----------
    support_vectors_ : ndarray (n_support, n_features) float32
        Training rows kept as support vectors (``|β| > 0``).
    dual_coef_ : ndarray (n_support,) float32
        Dual coefficients ``β = α⁺ - α⁻`` aligned to ``support_vectors_``.
    intercept_ : float
        Regression intercept ``b``.
    n_support_ : int
        Number of support vectors.
    n_iter_ : int
        SMO passes actually run.
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
        eps: float = 0.1,
        tolerance: float = 1e-3,
        max_iter: int = 200,
        seed: int = 42,
    ) -> None:
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.c = c
        self.eps = eps
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.seed = seed
        self._model: _MetalSVR | None = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        X: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> "MetalSVR":
        """Fit the SVR with the GPU Gram matrix + host SMO.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or flat list (float32).
        y : ndarray of shape (n_samples,) float regression targets.
        n, d : optional explicit shape; inferred from ``np.asarray(X).shape``.
        """
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        flat = np.ascontiguousarray(Xa, dtype=np.float32).tobytes()
        y_arr = np.ascontiguousarray(y, dtype=np.float32).tobytes()

        model = _MetalSVR(
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
            self.c,
            self.eps,
            self.tolerance,
            self.max_iter,
            self.seed,
        )
        model.fit_bytes(flat, y_arr, n, d)
        self._model = model
        self.n_features_in_ = d

        self.support_vectors_ = np.array(
            model.support_vectors, dtype=np.float32
        ).reshape(model.support_count, d)
        self.dual_coef_ = np.array(model.dual_coef, dtype=np.float32)
        self.intercept_ = float(model.intercept)
        self.n_support_ = int(model.support_count)
        self.n_iter_ = int(model.n_iter)
        self.gamma_ = float(model.gamma)
        return self

    def predict(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> np.ndarray:
        """Predict the regression values ``f(X)`` of shape ``(n,)`` float32."""
        if self._model is None:
            raise RuntimeError("MetalSVR is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        raw = self._model.predict_bytes(
            np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), n, d
        )
        return np.array(raw, dtype=np.float32)

    def decision_function(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> np.ndarray:
        """Raw regression values ``f(X)`` (identical to ``predict``)."""
        if self._model is None:
            raise RuntimeError("MetalSVR is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        raw = self._model.decision_function_bytes(
            np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), n, d
        )
        return np.array(raw, dtype=np.float32)

    def score(
        self,
        X: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> float:
        """Return the R² coefficient of determination ``1 - SS_res/SS_tot``."""
        if self._model is None:
            raise RuntimeError("MetalSVR is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        y_arr = np.ascontiguousarray(y, dtype=np.float32).tobytes()
        return float(
            self._model.score_bytes(
                np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), y_arr, n, d
            )
        )


def metal_svr(
    data: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n: int,
    d: int,
    kernel: str = "rbf",
    gamma: float = 0.0,
    degree: float = 3.0,
    coef0: float = 0.0,
    c: float = 1.0,
    eps: float = 0.1,
    tolerance: float = 1e-3,
    max_iter: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, int, float, int]:
    """Fit an SVR on the GPU and return its model parameters.

    .. note::
       The functional API does not retain the fitted model for later
       prediction; use the ``MetalSVR`` class when you need
       ``predict`` / ``decision_function`` / ``score``.

    Returns
    -------
    support_vectors_ : ndarray (n_support, d) float32
    dual_coef_ : ndarray (n_support,) float32  (β = α⁺ - α⁻)
    intercept_ : float
    n_support_ : int
    gamma_ : float
    n_iter_ : int
    """
    flat = _as_bytes_f32(data)
    y_arr = _as_bytes_f32(y)
    sv, dual, intercept, count, g, iters = _metal_svr_fit(
        flat,
        y_arr,
        n,
        d,
        kernel,
        gamma,
        degree,
        coef0,
        c,
        eps,
        tolerance,
        max_iter,
        seed,
    )
    return (
        np.array(sv, dtype=np.float32).reshape(count, d),
        np.array(dual, dtype=np.float32),
        intercept,
        count,
        g,
        iters,
    )