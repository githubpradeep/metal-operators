"""metal_kmeans example: synthetic 3-cluster data in 2D + larger benchmark."""

import numpy as np
from metal_kmeans import metal_kmeans, MetalKMeans


def smoke_test():
    """Small 2D example with 3 well-separated clusters."""
    rng = np.random.RandomState(42)
    data = np.vstack([
        rng.randn(300, 2) + [0, 0],
        rng.randn(300, 2) + [5, 5],
        rng.randn(400, 2) + [10, 0],
    ]).astype(np.float32)
    n, d = data.shape

    # ── Functional API ──
    labels, centroids, n_iter, inertia = metal_kmeans(
        data.ravel().tolist(), n, d, n_clusters=3,
        max_iterations=50, tolerance=1e-4, seed=42,
    )
    print("[functional]  n_iter={}  inertia={:.2f}".format(n_iter, inertia))
    print("  cluster sizes:", np.bincount(labels))
    print("  centroids:\n", np.array(centroids).reshape(3, d))

    # ── sklearn-style API ──
    km = MetalKMeans(n_clusters=3, max_iterations=50, tolerance=1e-4, seed=42)
    km.fit(data, n, d)  # accepts numpy array directly
    print("[sklearn]     n_iter_={}  inertia_={:.2f}".format(km.n_iter_, km.inertia_))
    print("  cluster sizes:", np.bincount(km.labels_))
    print("  cluster_centers_:\n", km.cluster_centers_)

    new_points = np.array([[1., 2.], [6., 6.], [9., 1.]], dtype=np.float32)
    pred = km.predict(new_points.ravel().tolist(), 3, 2)
    print("  predictions for new points:", pred)


def benchmark():
    """Larger shape to see GPU speed."""
    n, d, k = 100_000, 32, 64
    rng = np.random.RandomState(7)
    data = rng.randn(n, d).astype(np.float32)

    import time
    km = MetalKMeans(n_clusters=k, max_iterations=15, tolerance=1e-4, seed=7)
    t0 = time.perf_counter()
    km.fit(data, n, d)
    elapsed = time.perf_counter() - t0
    print("\n[benchmark]  {}×{} k={}  {:.0f} ms  inertia={:.2f}".format(
        n, d, k, elapsed * 1000, km.inertia_))


if __name__ == "__main__":
    print("=" * 50)
    print("metal_kmeans example")
    print("=" * 50)
    smoke_test()
    benchmark()
