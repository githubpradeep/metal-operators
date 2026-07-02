"""
metal_pca — Python wrappers for the Metal-accelerated PCA operator.
This package provides a scikit-learn-compatible API.

Usage:
    from metal_pca import MetalPCA
    pca = MetalPCA(n_components=2)
    pca.fit(data)  # data: numpy array or list-of-lists
    transformed = pca.transform(data)
    components = pca.components_
"""

import numpy as np
from metal_kmeans._native import MetalPCA as _MetalPCA


class MetalPCA:
    """PCA with Metal GPU acceleration.

    Parameters
    ----------
    n_components : int
        Number of principal components to compute.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self._model = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_features_ = None

    def fit(self, X, y=None):
        """Fit PCA to data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or list-of-lists
            Training data.
        y : ignored
            Present for sklearn Pipeline compatibility.

        Returns
        -------
        self : MetalPCA
        """
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        flat = X.ravel().tolist()

        self._model = _MetalPCA(self.n_components)
        self._model.fit(flat, n, d)

        comps = self._model.components
        ev = self._model.explained_variance
        evr = self._model.explained_variance_ratio
        self.n_features_ = d
        k_actual = len(ev)
        self.components_ = np.array(comps, dtype=np.float32).reshape(k_actual, d)
        self.explained_variance_ = np.array(ev, dtype=np.float32)
        self.explained_variance_ratio_ = np.array(evr, dtype=np.float32)
        try:
            self.singular_values_ = np.array(self._model.singular_values, dtype=np.float32)
        except Exception:
            self.singular_values_ = None
        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.n_features_in_ = d

        return self

    def transform(self, X):
        """Project data onto principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or list-of-lists
            Data to transform.

        Returns
        -------
        transformed : ndarray of shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        flat = X.ravel().tolist()
        result = self._model.transform(flat, n, d)
        k = len(self.explained_variance_)
        return np.array(result, dtype=np.float32).reshape(n, k)

    def fit_transform(self, X, y=None):
        """Fit PCA and project data in one step.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or list-of-lists
            Training data.
        y : ignored
            Present for sklearn Pipeline compatibility.

        Returns
        -------
        transformed : ndarray of shape (n_samples, n_components)
        """
        self.fit(X, y)
        return self.transform(X)
