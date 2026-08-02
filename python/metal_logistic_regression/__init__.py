"""metal_logistic_regression — GPU-accelerated binary logistic regression via Apple Metal.

Provides a functional API (``metal_logistic_regression``) and a sklearn-style
class (``MetalLogisticRegression``), mirroring the interface of sklearn's
``LogisticRegression``.

Usage::

    from metal_logistic_regression import metal_logistic_regression, MetalLogisticRegression

    # Functional API — returns (weights, bias, n_epochs, final_loss)
    weights, bias, n_epochs, final_loss = metal_logistic_regression(
        data, y, n, d, c=1.0, learning_rate=0.01, max_epochs=100, seed=42
    )

    # sklearn-style API
    clf = MetalLogisticRegression(c=1.0, max_epochs=100, seed=42)
    clf.fit(data, y, n, d)
    proba = clf.predict_proba(new_data, n_new, d)
    preds = clf.predict(new_data, n_new, d)
    acc = clf.score(data, y, n, d)

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of shape
``(n, d)`` in row-major order; ``y`` must be binary labels in ``{0.0, 1.0}``.

Training uses full-batch L-BFGS (m = 10): a fused forward/backward/loss Metal
kernel (naive / simdgroup / split-D variants) evaluates the whole dataset in
one launch per iteration, with the L-BFGS update computed on the host.

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalLogisticRegression as _MetalLogisticRegression
from metal_kmeans._native import (
    metal_logistic_regression_fit_bytes as _metal_logistic_regression_fit,
)

__all__ = ["MetalLogisticRegression", "metal_logistic_regression"]


class MetalLogisticRegression:
    """sklearn-style binary logistic regression using GPU-accelerated Metal kernels.

    Parameters
    ----------
    c : float, optional
        Inverse regularization strength; larger ``c`` = weaker regularization
        (default 1.0).
    learning_rate : float, optional
        Kept for API symmetry with flashlib; unused by the L-BFGS optimizer
        (default 0.01).
    momentum : float, optional
        Kept for API symmetry with flashlib; unused by the L-BFGS optimizer
        (default 0.9).
    max_epochs : int, optional
        Maximum number of L-BFGS iterations (default 100).
    batch_size : int, optional
        Kept for API symmetry; L-BFGS is full-batch (default 256).
    tol : float, optional
        Convergence tolerance on gradient sup-norm (default 1e-4).
    seed : int, optional
        Kept for API symmetry; the optimizer is deterministic (default 42).
    """

    def __init__(
        self,
        c: float = 1.0,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        max_epochs: int = 100,
        batch_size: int = 256,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self._d = 0
        self._inner = _MetalLogisticRegression(
            c, learning_rate, momentum, max_epochs, batch_size, tol, seed
        )

    def fit(
        self,
        data: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int,
        d: int,
    ) -> MetalLogisticRegression:
        """Fit the model to *data* with binary labels *y*.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` features as float32.
        y : ndarray | list[float]
            Binary labels of shape ``(n,)`` with values in ``{0.0, 1.0}``.
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

    def predict_proba(
        self, data: np.ndarray | list[float], n: int, d: int
    ) -> np.ndarray:
        """Predict class probabilities ``P(y=1 | x)`` for each sample.

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
        proba : np.ndarray of shape (n,) float32
            Probabilities in ``[0, 1]``.
        """
        arr = _as_bytes_f32(data)
        raw = self._inner.predict_proba_bytes(arr, n, d)
        return np.array(raw, dtype=np.float32)

    def predict(self, data: np.ndarray | list[float], n: int, d: int) -> np.ndarray:
        """Predict hard class labels (``0`` or ``1``).

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
        labels : np.ndarray of shape (n,) with dtype intp
        """
        arr = _as_bytes_f32(data)
        raw = self._inner.predict_bytes(arr, n, d)
        return np.array(raw, dtype=np.intp)

    def score(
        self,
        data: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int,
        d: int,
    ) -> float:
        """Return mean accuracy on the given data and labels.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` features as float32.
        y : ndarray | list[float]
            Binary labels of shape ``(n,)`` with values in ``{0.0, 1.0}``.
        n : int
            Number of samples.
        d : int
            Number of features.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples.
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
        """Bias term (sklearn-style name)."""
        return self._inner.intercept_

    @property
    def weights(self) -> np.ndarray:
        """Raw weight vector of shape ``(d,)`` float32."""
        return np.array(self._inner.weights, dtype=np.float32)

    @property
    def bias(self) -> float:
        """Raw bias term."""
        return self._inner.bias

    @property
    def n_epochs_(self) -> int:
        """Number of epochs actually run during the last fit."""
        return self._inner.n_epochs

    @property
    def final_loss_(self) -> float:
        """Loss at the end of the last fit (log-loss + L2 penalty)."""
        return self._inner.final_loss


def metal_logistic_regression(
    data: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n: int,
    d: int,
    c: float = 1.0,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    max_epochs: int = 100,
    batch_size: int = 256,
    tol: float = 1e-4,
    seed: int = 42,
) -> Tuple[np.ndarray, float, int, float]:
    """Fit binary logistic regression on the GPU and return the results.

    Parameters
    ----------
    data : ndarray | list[float]
        Flat row-major ``(n, d)`` features as float32.
    y : ndarray | list[float]
        Binary labels of shape ``(n,)`` with values in ``{0.0, 1.0}``.
    n : int
        Number of samples.
    d : int
        Number of features.
    c : float, optional
        Inverse regularization strength (default 1.0).
    learning_rate : float, optional
        Kept for API symmetry with flashlib; unused by the L-BFGS optimizer
        (default 0.01).
    momentum : float, optional
        Kept for API symmetry with flashlib; unused by the L-BFGS optimizer
        (default 0.9).
    max_epochs : int, optional
        Maximum number of L-BFGS iterations (default 100).
    batch_size : int, optional
        Kept for API symmetry; L-BFGS is full-batch (default 256).
    tol : float, optional
        Convergence tolerance on gradient sup-norm (default 1e-4).
    seed : int, optional
        Kept for API symmetry; the optimizer is deterministic (default 42).

    Returns
    -------
    weights : np.ndarray of shape (d,) float32
        Learned coefficients.
    bias : float
        Learned intercept.
    n_epochs : int
        L-BFGS iterations actually run (early-stopped on convergence).
    final_loss : float
        Log-loss + L2 penalty at the end of training.
    """
    arr = _as_bytes_f32(data)
    y_arr = _as_bytes_f32(y)
    raw_w, bias, n_epochs, final_loss = _metal_logistic_regression_fit(
        arr, y_arr, n, d, c, learning_rate, momentum, max_epochs, batch_size, tol, seed
    )
    weights = np.array(raw_w, dtype=np.float32)
    return weights, bias, n_epochs, final_loss


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    """Return raw little-endian float32 bytes of *data* (C-speed memcpy).

    Consumed by the ``_bytes`` bindings — avoiding the O(n) ``tolist()``
    roundtrip that the ``Vec<f32>`` methods pay (multi-second on 1M+ samples).
    """
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()
