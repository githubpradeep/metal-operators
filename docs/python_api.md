# Python API Reference

The `metal_kmeans` package provides GPU-accelerated KMeans clustering via Apple Metal.

```
pip install maturin
maturin develop   # from project root
```

## `metal_kmeans` (functional API)

```python
metal_kmeans(data, n, d, n_clusters,
             max_iterations=100, tolerance=1e-4, seed=42)
```

### Parameters

| Name | Type | Description |
|---|---|---|
| `data` | `np.ndarray[float32]` or `list[float]` | Flat row-major points: `data[i * d + j]` = point `i`, dim `j`. Shape `(n * d,)`. |
| `n` | `int` | Number of points. |
| `d` | `int` | Number of dimensions. |
| `n_clusters` | `int` | Number of clusters (K). |
| `max_iterations` | `int` | Lloyd iterations limit (default 100). |
| `tolerance` | `float` | Convergence threshold: max per-coordinate centroid shift (default 1e-4). |
| `seed` | `int` | RNG seed for k-means++ initialization. 0 = time-based (default 42). |

### Returns

| Name | Type | Shape | Description |
|---|---|---|---|
| `labels` | `np.ndarray[intp]` | `(n,)` | Cluster assignment for each point (0..K-1). |
| `centroids` | `np.ndarray[float32]` | `(K, d)` | Final cluster centers. |
| `n_iter` | `int` | — | Iterations executed. |
| `inertia` | `float` | — | Within-cluster sum of squared distances. |

### Errors

Raises `RuntimeError` if Metal is unavailable, shader compilation fails, or input parameters are invalid (e.g. `K > n`, data length mismatch).

---

## `MetalKMeans` (sklearn-style class)

```python
km = MetalKMeans(n_clusters, max_iterations=100, tolerance=1e-4, seed=42)
km.fit(data, n, d)
km.predict(data, n, d)
```

### Constructor

| Argument | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | `int` | — | Number of clusters. |
| `max_iterations` | `int` | `100` | Lloyd iteration limit. |
| `tolerance` | `float` | `1e-4` | Convergence threshold. |
| `seed` | `int` | `42` | k-means++ seed. |

### Methods

#### `fit(data, n, d) -> MetalKMeans`

Run Lloyd's algorithm. Returns `self` for chaining.

| Param | Type | Description |
|---|---|---|
| `data` | `np.ndarray[float32]` or `list[float]` | Flat row-major points. |
| `n` | `int` | Number of points. |
| `d` | `int` | Number of dimensions. |

#### `predict(data, n, d) -> np.ndarray[intp]`

Assign each point in `data` to the nearest fitted centroid. Does **not** update the model.

| Param | Type | Description |
|---|---|---|
| `data` | `np.ndarray[float32]` or `list[float]` | Flat row-major points. |
| `n` | `int` | Number of points. |
| `d` | `int` | Number of dimensions. |

Returns `labels` array of shape `(n,)` with dtype `intp`.

### Properties

| Property | Type | Shape | Description |
|---|---|---|---|
| `cluster_centers_` | `np.ndarray[float32]` | `(K, d)` | Final centroids from last fit. |
| `labels_` | `np.ndarray[intp]` | `(n,)` | Per-point labels from last fit. |
| `inertia_` | `float` | — | Sum of squared distances (last fit). |
| `n_iter_` | `int` | — | Iterations used (last fit). |

### Example

```python
from metal_kmeans import MetalKMeans
import numpy as np

data = np.random.randn(10000, 32).astype(np.float32)
n, d = data.shape

km = MetalKMeans(n_clusters=8, max_iterations=15)
km.fit(data, n, d)
print(f"inertia={km.inertia_:.2f}  n_iter={km.n_iter_}")

new_points = np.random.randn(500, 32).astype(np.float32)
labels = km.predict(new_points, 500, d)
print(np.bincount(labels))
```

---

## Data layout

`data` must be **flat row-major**:

```
data = [x0_0, x0_1, ..., x0_{d-1},
        x1_0, x1_1, ..., x1_{d-1},
        ...]
```

There is no copy if `data` is a `np.ndarray[float32]` — the ravel is efficient. Other dtypes cause an `astype` copy.

---

## Errors

All GPU errors surface as Python `RuntimeError`:

- `"No Metal device found"` — non-Apple hardware or macOS VM.
- `"Shader compilation failed: ..."` — Metal shader syntax error (unexpected on released builds).
- `"KMeans fit failed: ..."` — invalid parameters (`k > n`, `d == 0`, data length mismatch).

---

## Performance notes

- **First call cold**: ~20 ms per kernel (Metal deferred compilation). Subsequent calls reuse pipeline caches — <1 ms warm for centroid, ~0.3–6 ms for assign depending on shape.
- **`MetalContext` is shared** across all calls via a process-global `OnceLock`. First call initialises it; later calls are zero-overhead.
- **Kernel picker** selects Naive / Simdgroup / SimdgroupC16 / Split-D at runtime based on `n`, `d`, `k`, and shared-memory budget.
- **Centroid update** runs on GPU when `(K * d + K) * 4 <= 32768` bytes; otherwise falls back to CPU.
