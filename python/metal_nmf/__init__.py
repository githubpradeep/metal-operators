"""
metal_nmf — Python wrappers for the Metal-accelerated NMF operator.
This package provides a scikit-learn-compatible API.

Usage:
    from metal_nmf import MetalNMF
    model = MetalNMF(n_components=4)
    W = model.fit_transform(data)   # data: non-negative numpy array
    H = model.components_           # (n_components, n_features)
"""

import numpy as np
from metal_kmeans._native import MetalNMF as _MetalNMF


class MetalNMF:
    """Non-negative Matrix Factorization with Metal GPU acceleration.

    Factorizes a non-negative matrix X (n_samples, n_features) into the
    product X ≈ W · H, where H = components_ has shape
    (n_components, n_features) and W is the coefficient matrix.

    Parameters
    ----------
    n_components : int
        Number of latent components.
    max_iterations : int
        Maximum number of multiplicative-update iterations.
    tolerance : float
        Convergence tolerance on the reconstruction error change.
    seed : int
        Random seed for initialization.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The factor matrix H.
    reconstruction_err_ : float
        Frobenius norm of X - W·H after the final iteration.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, n_components=2, max_iterations=200, tolerance=1e-4, seed=42):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        self._model = None
        self.components_ = None
        self.reconstruction_err_ = None
        self.n_iter_ = None

    def fit(self, X, y=None):
        """Fit NMF to non-negative data *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or list-of-lists
            Non-negative training data.
        y : ignored
            Present for sklearn Pipeline compatibility.

        Returns
        -------
        self : MetalNMF
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        if (X < 0).any():
            raise ValueError("Negative values in data passed to MetalNMF.fit")
        n, d = X.shape
        flat = np.ascontiguousarray(X, dtype=np.float32).tobytes()

        self._model = _MetalNMF(self.n_components, self.max_iterations,
                                self.tolerance, self.seed)
        self._model.fit_bytes(flat, n, d)

        k = self.n_components
        self.components_ = np.array(self._model.components, dtype=np.float32).reshape(k, d)
        self.reconstruction_err_ = float(self._model.reconstruction_error)
        self.n_iter_ = int(self._model.n_iter)
        return self

    def transform(self, X):
        """Project new data onto the learned components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or list-of-lists
            Non-negative data.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Coefficient matrix for *X*.
        """
        if self._model is None or self.components_ is None:
            raise RuntimeError("This MetalNMF instance is not fitted yet; call fit first")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        if (X < 0).any():
            raise ValueError("Negative values in data passed to MetalNMF.transform")
        n, d = X.shape
        flat = np.ascontiguousarray(X, dtype=np.float32).tobytes()
        result = self._model.transform_bytes(flat, n, d)
        return np.array(result, dtype=np.float32).reshape(n, self.n_components)

    def fit_transform(self, X, y=None):
        """Fit NMF and return the coefficient matrix W for *X*."""
        self.fit(X, y)
        return np.array(self._model.coeff, dtype=np.float32).reshape(-1, self.n_components)
