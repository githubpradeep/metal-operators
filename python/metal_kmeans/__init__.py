"""metal_kmeans — GPU-accelerated clustering and nearest-neighbour search via Apple Metal.

Provides KMeans (``MetalKMeans`` / ``metal_kmeans``) and KNN (``MetalKNeighbors`` /
``metal_kneighbors``), mirroring the interface of flashlib_.

Provides both a functional API (``metal_kmeans``) and an sklearn-style
class (``MetalKMeans``), mirroring the interface of flashlib_.

Usage::

    from metal_kmeans import metal_kmeans, MetalKMeans

    # Functional API — returns (labels, centroids, n_iter, inertia)
    labels, centroids, n_iter, inertia = metal_kmeans(
        data, n, d, n_clusters=3, max_iterations=50, tolerance=1e-4, seed=42
    )

    # sklearn-style API
    km = MetalKMeans(n_clusters=3, max_iterations=50, tolerance=1e-4, seed=42)
    km.fit(data, n, d)
    labels = km.predict(new_data, new_n, d)  # uses fitted centroids

``data`` must be a flat ``list[float]`` or ``numpy.ndarray[float32]`` of
shape ``(n, d)`` in row-major order.

Startup note: the first call compiles Metal shaders (~20 ms/kernel); subsequent
calls reuse the cached pipeline state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._native import MetalKMeans as _MetalKMeans
from ._native import metal_kmeans_fit as _metal_kmeans_fit
from ._native import MetalKNeighbors as _MetalKNeighbors
from ._native import metal_kneighbors as _metal_kneighbors

__all__ = ["MetalKMeans", "metal_kmeans", "MetalKNeighbors", "metal_kneighbors"]


class MetalKMeans:
    """sklearn-style KMeans using GPU-accelerated Apple Metal kernels.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iterations : int, optional
        Maximum Lloyd iterations (default 100).
    tolerance : float, optional
        Convergence threshold on maximum centroid shift (default 1e-4).
    seed : int, optional
        RNG seed for k-means++ initialization (default 42).
    """

    def __init__(
        self,
        n_clusters: int,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self._n_clusters = n_clusters
        self._d = 0
        self._inner = _MetalKMeans(n_clusters, max_iterations, tolerance, seed)
    
    def fit(self, data: np.ndarray | list[float], n: int, d: int) -> MetalKMeans:
        """Fit KMeans to *data*.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` points as float32.
        n : int
            Number of points.
        d : int
            Number of dimensions.

        Returns
        -------
        self
        """
        arr = _to_vec_f32(data)
        self._inner.fit(arr, n, d)
        self._d = d
        return self

    def predict(self, data: np.ndarray | list[float], n: int, d: int) -> np.ndarray:
        """Assign each point in *data* to the nearest fitted centroid.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` points as float32.
        n : int
            Number of points.
        d : int
            Number of dimensions.

        Returns
        -------
        labels : np.ndarray of shape (n,) with dtype intp
        """
        arr = _to_vec_f32(data)
        raw = self._inner.predict(arr, n, d)
        return np.array(raw, dtype=np.intp)

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Centroids as ``(n_clusters, d)`` float32 array."""
        raw = self._inner.centroids
        k = len(raw) // self._d  # infer (k*d) / d
        return np.array(raw, dtype=np.float32).reshape(k, self._d)

    @property
    def labels_(self) -> np.ndarray:
        """Per-point cluster labels from the last fit, shape ``(n,)`` intp."""
        return np.array(self._inner.labels, dtype=np.intp)

    @property
    def inertia_(self) -> float:
        """Within-cluster sum of squared distances."""
        return self._inner.inertia

    @property
    def n_iter_(self) -> int:
        """Number of iterations run on the last fit."""
        return self._inner.n_iter


def metal_kmeans(
    data: np.ndarray | list[float],
    n: int,
    d: int,
    n_clusters: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Run KMeans clustering on the GPU and return results.

    Parameters
    ----------
    data : ndarray | list[float]
        Flat row-major ``(n, d)`` points as float32.
    n : int
        Number of points.
    d : int
        Number of dimensions.
    n_clusters : int
        Number of clusters.
    max_iterations : int, optional
        Maximum Lloyd iterations (default 100).
    tolerance : float, optional
        Convergence threshold (default 1e-4).
    seed : int, optional
        RNG seed for k-means++ (default 42).

    Returns
    -------
    labels : np.ndarray of shape (n,) intp
        Cluster assignment for each point.
    centroids : np.ndarray of shape (n_clusters, d) float32
        Final cluster centers.
    n_iter : int
        Iterations used.
    inertia : float
        Within-cluster sum of squared distances.
    """
    arr = _to_vec_f32(data)
    raw_labels, raw_centroids, n_iter, inertia = _metal_kmeans_fit(
        arr, n, d, n_clusters, max_iterations, tolerance, seed
    )
    labels = np.array(raw_labels, dtype=np.intp)
    centroids = np.array(raw_centroids, dtype=np.float32).reshape(n_clusters, d)
    return labels, centroids, n_iter, inertia


def _to_vec_f32(data: np.ndarray | list[float]) -> list[float]:
    if isinstance(data, np.ndarray):
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        return data.ravel(order="C").tolist()
    return list(data)


# ── KNN ─────────────────────────────────────────────────────────

class MetalKNeighbors:
    """sklearn-style K-Nearest Neighbors using GPU-accelerated Apple Metal kernels.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of neighbors to retrieve (default 5).
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        self._n_neighbors = n_neighbors
        self._d = 0
        self._inner = _MetalKNeighbors(n_neighbors)

    def fit(self, data: np.ndarray | list[float], n: int, d: int) -> MetalKNeighbors:
        """Fit the corpus (database) of points.

        Parameters
        ----------
        data : ndarray | list[float]
            Flat row-major ``(n, d)`` points as float32.
        n : int
            Number of corpus points.
        d : int
            Number of dimensions.

        Returns
        -------
        self
        """
        arr = _to_vec_f32(data)
        self._inner.fit(arr, n, d)
        self._d = d
        return self

    def kneighbors(
        self, queries: np.ndarray | list[float], nq: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the K nearest neighbours of each query point.

        Parameters
        ----------
        queries : ndarray | list[float]
            Flat row-major ``(nq, d)`` query points as float32.
        nq : int
            Number of query points.

        Returns
        -------
        distances : np.ndarray of shape (nq, n_neighbors) float32
            Squared Euclidean distances to neighbours.
        indices : np.ndarray of shape (nq, n_neighbors) int64
            Indices of neighbours in the corpus.
        """
        arr = _to_vec_f32(queries)
        raw_d, raw_i = self._inner.kneighbors(arr, nq)
        k = self._n_neighbors
        distances = np.array(raw_d, dtype=np.float32).reshape(nq, k)
        indices = np.array(raw_i, dtype=np.intp).reshape(nq, k)
        return distances, indices


def metal_kneighbors(
    corpus: np.ndarray | list[float],
    n_corpus: int,
    d: int,
    queries: np.ndarray | list[float],
    n_queries: int,
    n_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the K nearest neighbours of each query point against a corpus.

    Parameters
    ----------
    corpus : ndarray | list[float]
        Flat row-major ``(n_corpus, d)`` corpus points as float32.
    n_corpus : int
        Number of corpus points.
    d : int
        Number of dimensions.
    queries : ndarray | list[float]
        Flat row-major ``(n_queries, d)`` query points as float32.
    n_queries : int
        Number of query points.
    n_neighbors : int, optional
        Number of neighbours (default 5).

    Returns
    -------
    distances : np.ndarray of shape (n_queries, n_neighbors) float32
        Squared Euclidean distances to neighbours.
    indices : np.ndarray of shape (n_queries, n_neighbors) int64
        Indices of neighbours in the corpus.
    """
    c_arr = _to_vec_f32(corpus)
    q_arr = _to_vec_f32(queries)
    raw_d, raw_i = _metal_kneighbors(c_arr, n_corpus, d, q_arr, n_queries, n_neighbors)
    distances = np.array(raw_d, dtype=np.float32).reshape(n_queries, n_neighbors)
    indices = np.array(raw_i, dtype=np.intp).reshape(n_queries, n_neighbors)
    return distances, indices
