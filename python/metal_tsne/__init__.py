"""metal_tsne — Python wrapper for the Metal-accelerated t-SNE operator.

Provides a scikit-learn-compatible API mirroring ``sklearn.manifold.TSNE``
(exact, ``method="exact"``) for nonlinear dimensionality reduction / embedding.

Usage::

    from metal_tsne import MetalTSNE
    import numpy as np

    X = np.random.randn(300, 8).astype(np.float32)

    tsne = MetalTSNE(n_components=2, perplexity=30.0, n_iter=1000)
    Y = tsne.fit_transform(X)     # (300, 2) embedding, ndarray of float32
    kl = tsne.kl_divergence_      # final KL(P||Q)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from metal_kmeans._native import MetalTSNE as _MetalTSNE
from metal_kmeans._native import metal_tsne_fit_bytes as _metal_tsne_fit

__all__ = ["MetalTSNE", "metal_tsne"]


def _as_bytes_f32(data: np.ndarray | list[float]) -> bytes:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float32).tobytes()
    return np.asarray(data, dtype=np.float32).tobytes()


class MetalTSNE:
    """t-Distributed Stochastic Neighbor Embedding with Metal acceleration.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space (1..8).
    perplexity : float, default=30.0
        Target perplexity; must satisfy ``1 <= perplexity < n_samples``.
    learning_rate : float, default=200.0
        Gradient-descent step size.
    n_iter : int, default=1000
        Number of gradient descent iterations.
    early_exaggeration : float, default=12.0
        Multiplier applied to ``P`` during the first ``exaggeration_iter``
        iterations (>= 1.0; 1.0 disables).
    exaggeration_iter : int, default=250
        Number of early-exaggeration iterations.
    momentum : float, default=0.8
        Velocity (momentum) coefficient.
    seed : int, default=42
        Seed for the Gaussian embedding initialization (reproducible).
    min_grad_norm : float, default=1e-7
        Gradient-norm threshold for early stopping (0 disables).

    Attributes
    ----------
    embedding_ : ndarray (n_samples, n_components), float32
        The fitted low-dimensional embedding.
    n_iter_ : int
        Iterations actually run.
    kl_divergence_ : float
        Final KL(P‖Q) cost of the embedding.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        exaggeration_iter: int = 250,
        momentum: float = 0.8,
        seed: int = 42,
        min_grad_norm: float = 1e-7,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.exaggeration_iter = exaggeration_iter
        self.momentum = momentum
        self.seed = seed
        self.min_grad_norm = min_grad_norm
        self._model: _MetalTSNE | None = None

    def fit(
        self,
        X: np.ndarray | list[float],
        n: int | None = None,
        d: int | None = None,
    ) -> "MetalTSNE":
        """Fit the t-SNE embedding of *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or flat list.
        n, d : optional explicit shape; inferred from ``np.asarray(X).shape``.
        """
        Xa = np.asarray(X, dtype=np.float32)
        if n is None or d is None:
            n, d = Xa.shape
        flat = np.ascontiguousarray(Xa, dtype=np.float32).tobytes()

        model = _MetalTSNE(
            self.n_components,
            self.perplexity,
            self.learning_rate,
            self.n_iter,
            self.early_exaggeration,
            self.exaggeration_iter,
            self.momentum,
            self.seed,
            self.min_grad_norm,
        )
        model.fit_bytes(flat, n, d)
        self._model = model

        emb = np.array(model.embedding, dtype=np.float32).reshape(n, self.n_components)
        self.embedding_ = emb
        self.n_iter_ = model.n_iter
        self.kl_divergence_ = float(model.kl_divergence)
        self.n_features_in_ = d
        return self

    def fit_transform(
        self, X: np.ndarray | list[float], n: int | None = None, d: int | None = None
    ) -> np.ndarray:
        """Fit and return the embedding as an ``(n, n_components)`` ndarray."""
        self.fit(X, n, d)
        return self.embedding_


def metal_tsne(
    data: np.ndarray | list[float],
    n: int,
    d: int,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    early_exaggeration: float = 12.0,
    exaggeration_iter: int = 250,
    momentum: float = 0.8,
    seed: int = 42,
    min_grad_norm: float = 1e-7,
) -> Tuple[np.ndarray, int, float]:
    """Fit t-SNE on the GPU and return ``(embedding, n_iter, kl_divergence)``."""
    flat = _as_bytes_f32(data)
    emb, n_iter_, kl = _metal_tsne_fit(
        flat,
        n,
        d,
        n_components,
        perplexity,
        learning_rate,
        n_iter,
        early_exaggeration,
        exaggeration_iter,
        momentum,
        seed,
        min_grad_norm,
    )
    return (
        np.array(emb, dtype=np.float32).reshape(n, n_components),
        n_iter_,
        kl,
    )