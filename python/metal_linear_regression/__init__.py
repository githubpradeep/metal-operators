"""metal_linear_regression — GPU-accelerated linear regression via Apple Metal.

Provides a functional API (``metal_linear_regression``) and a sklearn-style
class (``MetalLinearRegression``), mirroring the interface of sklearn's
``LinearRegression`` / ``Ridge``.

Usage::

    from metal_linear_regression import metal_linear_regression, MetalLinearRegression

    # Functional API — returns (weights, bias, n_iter, final_loss)
    weights, bias, n_iter, final_loss = metal_linear_regression(
        data, y, n, d, alpha=0.0, fit_intercept=True
    )

    # sklearn-style API
    reg = MetalLinearRegression(alpha=0.0, fit_intercept=True)
    reg.fit(data, y, n, d)
    preds = reg.predict(new_data, n_new, d)
    r2 = reg.score(data, y, n, d)

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of shape
``(n, d)`` in row-major order; ``y`` are continuous targets of shape ``(n,)``.

Training solves the normal equations in closed form: two Metal kernels
stream the dataset once each to build the augmented Gram matrix
(``[XᵀX | Xᵀ·1; 1ᵀ·X | n]``) and right-hand side (``[Xᵀy; Σy]``), a tiny
deterministic reduction combines per-threadgroup partials, and the
(d+1)×(d+1) system is solved on the host with Gaussian elimination +
partial pivoting. ``alpha > 0`` adds L2 ridge regularization on the
coefficients (sklearn ``Ridge`` convention).

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalLinearRegression as _MetalLinearRegression
from metal_kmeans._native import (
    metal_linear_regression_fit_bytes as _metal_linear_regression_fit,
)

__all__ = ["MetalLinearRegression", "metal_linear_regression"]


class MetalLinearRegression:
    """sklearn-style linear regression using GPU-accelerated Metal kernels.

    Parameters
    ----------
    alpha : float, optional
        L2 regularization strength on the coefficients; ``0.0`` = plain
        ordinary least squares (default 0.0).
    fit_intercept : bool, optional
        Whether to fit an intercept (bias) term (default True).
    max_iterations : int, optional
        Kept for API symmetry with flashlib; unused by the closed-form
        solver (default 100).
    tol : float, optional
        Kept for API symmetry; unused by the closed-form solver
        (default 1e-4).
    seed : int, optional
        Kept for API symmetry; the solver is deterministic (default 42).
    """

    def __init__(
        self,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        max_iterations: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self._d = 0
        self._inner = _MetalLinearRegression(
            alpha, fit_intercept, max_iterations, tol, seed
        )

    def fit(
        self,
        data: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int,
        d: int,
    ) -> MetalLinearRegression:
        """Fit the model to *data* with continuous targets *y*.

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
        """Solver iterations (1 = closed-form solve)."""
        return self._inner.n_iter

    @property
    def final_loss(self) -> float:
        """Mean squared error on the training data after fit."""
        return self._inner.final_loss


def metal_linear_regression(
    data: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n: int,
    d: int,
    alpha: float = 0.0,
    fit_intercept: bool = True,
    max_iterations: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> Tuple[np.ndarray, float, int, float]:
    """Fit linear regression on the GPU and return the results.

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
        L2 regularization strength on the coefficients; ``0.0`` = ordinary
        least squares (default 0.0).
    fit_intercept : bool, optional
        Whether to fit an intercept (bias) term (default True).
    max_iterations : int, optional
        Kept for API symmetry with flashlib; unused (default 100).
    tol : float, optional
        Kept for API symmetry; unused (default 1e-4).
    seed : int, optional
        Kept for API symmetry; the solver is deterministic (default 42).

    Returns
    -------
    weights : np.ndarray of shape (d,) float32
        Learned coefficients.
    bias : float
        Learned intercept.
    n_iter : int
        Solver iterations (1 = closed-form solve).
    final_loss : float
        Mean squared error on the training data.
    """
    arr = _as_bytes_f32(data)
    y_arr = _as_bytes_f32(y)
    raw_w, bias, n_iter, final_loss = _metal_linear_regression_fit(
        arr, y_arr, n, d, alpha, fit_intercept, max_iterations, tol, seed
    )
    weights = np.array(raw_w, dtype=np.float32)
    return weights, bias, n_iter, final_loss


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    """Return raw little-endian float32 bytes of *data* (C-speed memcpy).

    Consumed by the ``_bytes`` bindings — avoiding the O(n) ``tolist()``
    roundtrip that the ``Vec<f32>`` methods pay (multi-second on 1M+ samples).
    """
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()
