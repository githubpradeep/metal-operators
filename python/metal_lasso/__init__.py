"""metal_lasso — GPU-accelerated Lasso regression (L1-regularized) via Apple Metal.

Provides a functional API (``metal_lasso``) and a sklearn-style class
(``MetalLasso``), mirroring the interface of sklearn's ``linear_model.Lasso``.

Usage::

    from metal_lasso import metal_lasso, MetalLasso

    # Functional API — returns (weights, bias, n_iter, converged, final_loss)
    weights, intercept, n_iter, converged, mse = metal_lasso(
        data, y, n, d, alpha=1.0, fit_intercept=True
    )

    # sklearn-style API
    lasso = MetalLasso(alpha=1.0, fit_intercept=True)
    lasso.fit(data, y, n, d)
    preds = lasso.predict(new_data, n_new, d)
    r2 = lasso.score(data, y, n, d)

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of shape
``(n, d)`` in row-major order; ``y`` are continuous targets of shape ``(n,)``.

Training minimizes ``0.5·‖Xw − y‖² + alpha·‖w‖₁``. The heavy stage — building
the augmented Gram system ``[XᵀX | Xᵀ·1; 1ᵀ·X | n]`` and right-hand side
``[Xᵀy; Σy]`` — runs on the GPU in a single command buffer, then host
coordinate descent (Gauss-Seidel) sweeps the small ``(d+1)²`` Gram with
soft-thresholding updates until convergence. The intercept is never
penalized. ``alpha = 0`` reduces to ordinary least squares.

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalLasso as _MetalLasso
from metal_kmeans._native import metal_lasso_fit_bytes as _metal_lasso_fit

__all__ = ["MetalLasso", "metal_lasso"]


class MetalLasso:
    """sklearn-style Lasso regression using GPU-accelerated Metal kernels.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 penalty (default 1.0). Larger values
        drive more coefficients toward exactly zero.
    fit_intercept : bool, optional
        Whether to fit an intercept (bias) term (default True). The intercept
        is never penalized.
    max_iterations : int, optional
        Maximum number of coordinate-descent sweeps (default 1000).
    tol : float, optional
        Convergence tolerance: stop when a full sweep moves every coefficient
        by at most ``tol`` (default 1e-4).
    seed : int, optional
        Kept for API symmetry; the solver is deterministic (default 42).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iterations: int = 1000,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self._d = 0
        self._inner = _MetalLasso(alpha, fit_intercept, max_iterations, tol, seed)

    def fit(
        self,
        data: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int,
        d: int,
    ) -> MetalLasso:
        """Fit the Lasso model to *data* with continuous targets *y*.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` features as float32.
        y : ndarray | list[float]
            Continuous targets of shape ``(n,)``.
        n : int
            Number of samples.
        d : int
            Number of features.

        Returns
        -------
        self
        """
        arr = _as_bytes_f32(data)
        y_arr = _as_bytes_f32(y)
        self._inner.fit_bytes(arr, y_arr, n, d)
        self._d = d
        return self

    def predict(self, data: np.ndarray | list[float], n: int, d: int) -> np.ndarray:
        """Predict continuous targets for *data*.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` features as float32.
        n : int
            Number of samples.
        d : int
            Number of features.

        Returns
        -------
        preds : np.ndarray of shape (n,) float32
        """
        arr = _as_bytes_f32(data)
        raw = self._inner.predict_bytes(arr, n, d)
        return np.array(raw, dtype=np.float32)

    def score(
        self,
        data: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int,
        d: int,
    ) -> float:
        """Return the coefficient of determination (R²) of the prediction.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` features as float32.
        y : ndarray | list[float]
            Continuous targets of shape ``(n,)``.
        n : int
            Number of samples.
        d : int
            Number of features.

        Returns
        -------
        r2 : float
            Best possible score is 1.0 (lower is worse).
        """
        arr = _as_bytes_f32(data)
        y_arr = _as_bytes_f32(y)
        return self._inner.score_bytes(arr, y_arr, n, d)

    @property
    def coef_(self) -> np.ndarray:
        """Coefficient vector of shape ``(d,)`` float32 (sklearn-style name)."""
        return np.array(self._inner.coef_, dtype=np.float32)

    @property
    def intercept_(self) -> float:
        """Intercept term (sklearn-style name)."""
        return self._inner.intercept_

    @property
    def weights(self) -> np.ndarray:
        """Raw coefficient vector of shape ``(d,)`` float32."""
        return np.array(self._inner.weights, dtype=np.float32)

    @property
    def bias(self) -> float:
        """Raw intercept term."""
        return self._inner.bias

    @property
    def n_iter(self) -> int:
        """Number of coordinate-descent sweeps performed at fit time."""
        return self._inner.n_iter

    @property
    def converged(self) -> bool:
        """Whether the solver converged within ``max_iterations`` sweeps."""
        return self._inner.converged

    @property
    def final_loss(self) -> float:
        """Mean squared error on the training data after fit."""
        return self._inner.final_loss


def metal_lasso(
    data: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n: int,
    d: int,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    max_iterations: int = 1000,
    tol: float = 1e-4,
    seed: int = 42,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """Fit Lasso regression on the GPU and return the results.

    Parameters
    ----------
    data : ndarray | list[float]
        Flat row-major ``(n, d)`` features as float32.
    y : ndarray | list[float]
        Continuous targets of shape ``(n,)``.
    n : int
        Number of samples.
    d : int
        Number of features.
    alpha : float, optional
        Regularization strength for the L1 penalty (default 1.0).
    fit_intercept : bool, optional
        Whether to fit an intercept (default True).
    max_iterations : int, optional
        Maximum coordinate-descent sweeps (default 1000).
    tol : float, optional
        Convergence tolerance (default 1e-4).
    seed : int, optional
        Random seed (unused; solver is deterministic) (default 42).

    Returns
    -------
    weights : np.ndarray of shape (d,) float32
        Learned coefficients.
    intercept : float
        Learned intercept.
    n_iter : int
        Coordinate-descent sweeps performed.
    converged : bool
        Whether the solver converged within ``tol``.
    final_loss : float
        Mean squared error on the training data.
    """
    arr = _as_bytes_f32(data)
    y_arr = _as_bytes_f32(y)
    raw_w, bias, n_iter, converged, final_loss = _metal_lasso_fit(
        arr, y_arr, n, d, alpha, fit_intercept, max_iterations, tol, seed
    )
    weights = np.array(raw_w, dtype=np.float32)
    return weights, bias, n_iter, converged, final_loss


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    """Return raw little-endian float32 bytes of *data* (C-speed memcpy)."""
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()