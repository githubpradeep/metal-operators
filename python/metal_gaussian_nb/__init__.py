"""metal_gaussian_nb — GPU-accelerated Gaussian Naive Bayes via Apple Metal.

Provides a functional API (``metal_gaussian_nb_fit``) and a sklearn-style class
(``MetalGaussianNB``), mirroring the interface of sklearn's ``GaussianNB``.

Usage::

    from metal_gaussian_nb import metal_gaussian_nb_fit, MetalGaussianNB

    # Functional API — returns (theta_, var_, class_prior_)
    theta, var, prior = metal_gaussian_nb_fit(data, y, n, d, n_classes)

    # sklearn-style API
    clf = MetalGaussianNB(var_smoothing=1e-9)
    clf.fit(data, y, n, d, n_classes)
    proba = clf.predict_proba(new_data, n_new, d)   # (n_new, k)
    preds = clf.predict(new_data, n_new, d)         # (n_new,)
    acc = clf.score(test_data, test_labels, n_test, d)

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of shape
``(n, d)`` in row-major order; ``y`` must be class labels as integers in
``[0, n_classes)``.

Fit is a GPU reduction pass (per-class feature sum / sum-of-squares) with a
single CPU-GPU sync; predict is one GPU launch computing per-class log
posteriors, with the argmax / softmax evaluated on the host.

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalGaussianNB as _MetalGaussianNB
from metal_kmeans._native import metal_gaussian_nb_fit_bytes as _metal_gnb_fit

__all__ = ["MetalGaussianNB", "metal_gaussian_nb_fit"]


class MetalGaussianNB:
    """sklearn-style Gaussian Naive Bayes using GPU-accelerated Metal kernels.

    Parameters
    ----------
    var_smoothing : float, optional
        Fraction of the largest per-class feature variance added to every
        variance for numerical stability (default 1e-9).
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self._d = 0
        self._k = 0
        self._inner = _MetalGaussianNB(var_smoothing)

    def fit(
        self,
        data: np.ndarray | list[float],
        y: np.ndarray | list[float],
        n: int,
        d: int,
        n_classes: int,
    ) -> MetalGaussianNB:
        """Fit the model to *data* with integer class labels *y* in [0, n_classes).

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` features as float32.
        y : ndarray | list[float]
            Class labels of shape ``(n,)`` as integers in ``[0, n_classes)``.
        n : int
            Number of samples.
        d : int
            Number of features.
        n_classes : int
            Number of classes (>= 2).

        Returns
        -------
        self
        """
        arr = _as_bytes_f32(data)
        y_arr = _as_bytes_f32(y)
        self._inner.fit_bytes(arr, y_arr, n, d, n_classes)
        self._d = d
        self._k = n_classes
        return self

    def predict_log_proba(
        self, data: np.ndarray | list[float], n: int, d: int
    ) -> np.ndarray:
        """Predict normalized log-probabilities ``log P(y=c | x)`` per sample.

        Returns an array of shape ``(n, n_classes)`` float32.
        """
        arr = _as_bytes_f32(data)
        raw = self._inner.predict_log_proba_bytes(arr, n, d)
        return np.array(raw, dtype=np.float32).reshape(n, self._k)

    def predict_proba(
        self, data: np.ndarray | list[float], n: int, d: int
    ) -> np.ndarray:
        """Predict class probabilities ``P(y=c | x)`` per sample.

        Returns an array of shape ``(n, n_classes)`` float32, rows sum to 1.
        """
        arr = _as_bytes_f32(data)
        raw = self._inner.predict_proba_bytes(arr, n, d)
        return np.array(raw, dtype=np.float32).reshape(n, self._k)

    def predict(self, data: np.ndarray | list[float], n: int, d: int) -> np.ndarray:
        """Predict hard class labels (argmax over classes).

        Returns labels of shape ``(n,)`` with dtype intp.
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
        """Return mean accuracy on the given data and labels."""
        arr = _as_bytes_f32(data)
        y_arr = _as_bytes_f32(y)
        return self._inner.score_bytes(arr, y_arr, n, d)

    @property
    def theta_(self) -> np.ndarray:
        """Per-class feature means of shape ``(n_classes, d)`` float32."""
        return np.array(self._inner.theta_, dtype=np.float32).reshape(self._k, self._d)

    @property
    def var_(self) -> np.ndarray:
        """Per-class feature variances of shape ``(n_classes, d)`` float32."""
        return np.array(self._inner.var_, dtype=np.float32).reshape(self._k, self._d)

    @property
    def class_prior_(self) -> np.ndarray:
        """Empirical class priors of shape ``(n_classes,)`` float32."""
        return np.array(self._inner.class_prior_, dtype=np.float32)


def metal_gaussian_nb_fit(
    data: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n: int,
    d: int,
    n_classes: int,
    var_smoothing: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian Naive Bayes model and return its learned statistics.

    Parameters
    ----------
    data : ndarray | list[float]
        Flat row-major ``(n, d)`` features as float32.
    y : ndarray | list[float]
        Class labels of shape ``(n,)`` as integers in ``[0, n_classes)``.
    n : int
        Number of samples.
    d : int
        Number of features.
    n_classes : int
        Number of classes (>= 2).
    var_smoothing : float, optional
        Variance-smoothing fraction (default 1e-9).

    Returns
    -------
    theta_ : np.ndarray of shape (n_classes, d) float32
        Per-class feature means.
    var_ : np.ndarray of shape (n_classes, d) float32
        Per-class feature variances (after smoothing).
    class_prior_ : np.ndarray of shape (n_classes,) float32
        Empirical priors.
    """
    arr = _as_bytes_f32(data)
    y_arr = _as_bytes_f32(y)
    raw_theta, raw_var, raw_prior = _metal_gnb_fit(arr, y_arr, n, d, n_classes, var_smoothing)
    theta = np.array(raw_theta, dtype=np.float32).reshape(n_classes, d)
    var = np.array(raw_var, dtype=np.float32).reshape(n_classes, d)
    prior = np.array(raw_prior, dtype=np.float32)
    return theta, var, prior


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    """Return raw little-endian float32 bytes of *data* (C-speed memcpy).

    Consumed by the ``_bytes`` bindings — avoiding the O(n) ``tolist()``
    roundtrip that the ``Vec<f32>`` methods pay (multi-second on 1M+ samples).
    """
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()