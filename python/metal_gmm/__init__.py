"""metal_gmm — GPU-accelerated Gaussian Mixture Model (GMM) via Apple Metal.

Provides a functional API (``metal_gmm``) and a scikit-learn-compatible class
(``MetalGMM``), mirroring ``sklearn.mixture.GaussianMixture`` (full covariance,
EM, k-means++ init).

Usage::

    from metal_gmm import MetalGMM, metal_gmm
    import numpy as np

    X = np.random.randn(600, 4).astype(np.float32)
    X[:200, 0] += 6.0      # three well-separated blobs
    X[200:400, 1] += 6.0

    # sklearn-style API
    gmm = MetalGMM(n_components=3, seed=42)
    gmm.fit(X)
    labels = gmm.predict(X)          # (n,) hard component assignment
    proba = gmm.predict_proba(X)     # (n, k) responsibilities
    ll = gmm.score(X)                # average log-likelihood

    # Functional API — returns (weights, means, covariances, responsibilities,
    # lower_bound, n_iter)
    w, means, covs, resp, lb, iters = metal_gmm(X, *X.shape, n_components=3)

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of shape
``(n, d)`` in row-major order. ``fit`` runs the EM loop with the per-iteration
E-step (the O(n·k·d²) log-likelihood matrix) on the GPU and the M-step on the
host; ``predict``/``predict_proba``/``score`` are each a single GPU launch.

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalGMM as _MetalGMM
from metal_kmeans._native import metal_gmm_fit_bytes as _metal_gmm_fit

__all__ = ["MetalGMM", "metal_gmm"]


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()


class MetalGMM:
    """sklearn-style Gaussian Mixture Model using GPU-accelerated Metal kernels.

    Parameters
    ----------
    n_components : int, default=3
        Number of mixture components.
    max_iterations : int, default=100
        Maximum number of EM iterations.
    tolerance : float, default=1e-3
        Convergence threshold on the change in average log-likelihood.
    seed : int, default=42
        Seed for the k-means++ initialization (reproducible).
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of every covariance.

    Attributes
    ----------
    weights_ : ndarray (n_components,), float32
        Fitted mixture weights (sum to 1).
    means_ : ndarray (n_components, n_features), float32
        Fitted component means.
    covariances_ : ndarray (n_components, n_features, n_features), float32
        Fitted full covariance matrices.
    responsibilities_ : ndarray (n_samples, n_components), float32
        Posterior component probabilities of the fitted data.
    lower_bound_ : float
        Final average log-likelihood lower bound.
    n_iter_ : int
        EM iterations actually run.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        seed: int = 42,
        reg_covar: float = 1e-6,
    ) -> None:
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        self.reg_covar = reg_covar
        self._model: _MetalGMM | None = None

    def fit(
        self,
        X: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> "MetalGMM":
        """Fit the mixture model to *X* with the EM algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or flat list.
        n, d : optional explicit shape; inferred from ``np.asarray(X).shape``.
        """
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        flat = np.ascontiguousarray(Xa, dtype=np.float32).tobytes()

        model = _MetalGMM(
            self.n_components,
            self.max_iterations,
            self.tolerance,
            self.seed,
            self.reg_covar,
        )
        model.fit_bytes(flat, n, d)
        self._model = model

        self.weights_ = np.array(model.weights, dtype=np.float32).reshape(self.n_components)
        self.means_ = np.array(model.means, dtype=np.float32).reshape(self.n_components, d)
        self.covariances_ = np.array(model.covariances, dtype=np.float32).reshape(
            self.n_components, d, d
        )
        self.responsibilities_ = np.array(model.responsibilities, dtype=np.float32).reshape(
            n, self.n_components
        )
        self.lower_bound_ = float(model.lower_bound)
        self.n_iter_ = model.n_iter
        self.n_features_in_ = d
        return self

    def fit_predict(
        self,
        X: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> np.ndarray:
        """Fit and return the hard component assignment (argmax responsibility)."""
        self.fit(X, n, d)
        return np.argmax(self.responsibilities_, axis=1).astype(np.intp)

    def predict(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> np.ndarray:
        """Predict the most likely component for each sample.

        Returns labels of shape ``(n,)`` with dtype intp.
        """
        if self._model is None:
            raise RuntimeError("MetalGMM is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        raw = self._model.predict_bytes(
            np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), n, d
        )
        return np.array(raw, dtype=np.intp)

    def predict_proba(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> np.ndarray:
        """Posterior component probabilities (responsibilities) per sample.

        Returns an array of shape ``(n, n_components)`` float32; rows sum to 1.
        """
        if self._model is None:
            raise RuntimeError("MetalGMM is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        raw = self._model.predict_proba_bytes(
            np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), n, d
        )
        return np.array(raw, dtype=np.float32).reshape(n, self.n_components)

    def score(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> float:
        """Average log-likelihood of *X* under the fitted model."""
        if self._model is None:
            raise RuntimeError("MetalGMM is not fitted; call fit() first.")
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        return float(
            self._model.score_bytes(
                np.ascontiguousarray(Xa, dtype=np.float32).tobytes(), n, d
            )
        )


def metal_gmm(
    data: np.ndarray | list[float],
    n: int,
    d: int,
    n_components: int = 3,
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    seed: int = 42,
    reg_covar: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Fit a GMM on the GPU and return
    ``(weights, means, covariances, responsibilities, lower_bound, n_iter)``.
    """
    flat = _as_bytes_f32(data)
    w, means, covs, resp, lb, iters = _metal_gmm_fit(
        flat,
        n,
        d,
        n_components,
        max_iterations,
        tolerance,
        seed,
        reg_covar,
    )
    return (
        np.array(w, dtype=np.float32).reshape(n_components),
        np.array(means, dtype=np.float32).reshape(n_components, d),
        np.array(covs, dtype=np.float32).reshape(n_components, d, d),
        np.array(resp, dtype=np.float32).reshape(n, n_components),
        lb,
        iters,
    )