"""
metal_dbscan — Python wrappers for the Metal-accelerated DBSCAN operator.
This package provides a scikit-learn-compatible API.

Usage:
    from metal_dbscan import MetalDBSCAN
    db = MetalDBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(data)  # data: numpy array or list-of-lists
    print(db.labels_)      # cluster index per sample (-1 = noise)
    print(db.n_clusters_)  # number of clusters found
"""

import numpy as np
from metal_kmeans._native import MetalDBSCAN as _MetalDBSCAN


class MetalDBSCAN:
    """DBSCAN density-based clustering with Metal GPU acceleration.

    Parameters
    ----------
    eps : float
        Maximum distance between two samples for one to be considered
        in the neighborhood of the other.
    min_samples : int
        Number of samples in a neighborhood for a point to be considered
        a core point.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index per sample; -1 indicates noise.
    n_clusters_ : int
        Number of clusters found (noise excluded).
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self._model = None
        self.labels_ = None
        self.n_clusters_ = 0

    def fit(self, X, y=None):
        """Perform DBSCAN clustering on *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or list-of-lists
            Training data.
        y : ignored
            Present for sklearn Pipeline compatibility.

        Returns
        -------
        self : MetalDBSCAN
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        n, d = X.shape
        flat = np.ascontiguousarray(X, dtype=np.float32).tobytes()

        self._model = _MetalDBSCAN(self.eps, self.min_samples)
        self._model.fit_bytes(flat, n, d)

        self.labels_ = np.array(self._model.labels, dtype=np.int64)
        self.n_clusters_ = int(self._model.n_clusters)
        return self

    def fit_predict(self, X, y=None):
        """Fit DBSCAN and return the cluster labels."""
        self.fit(X, y)
        return self.labels_
