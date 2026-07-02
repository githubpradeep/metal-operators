# Python API Reference

The `metal_kmeans` package provides GPU-accelerated KMeans clustering and
K-Nearest Neighbour search via Apple Metal.

The `metal_pca` package provides GPU-accelerated PCA via Apple Metal.

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

---

## `metal_kneighbors` (functional API)

```python
metal_kneighbors(corpus, n_corpus, d, queries, n_queries, n_neighbors=5)
```

### Parameters

| Name | Type | Description |
|---|---|---|
| `corpus` | `np.ndarray[float32]` or `list[float]` | Flat row-major corpus points: `corpus[i * d + j]` = point `i`, dim `j`. Shape `(n_corpus * d,)`. |
| `n_corpus` | `int` | Number of corpus points. |
| `d` | `int` | Number of dimensions. |
| `queries` | `np.ndarray[float32]` or `list[float]` | Flat row-major query points, shape `(n_queries * d,)`. |
| `n_queries` | `int` | Number of query points. |
| `n_neighbors` | `int` | Number of nearest neighbours to retrieve (default 5). |

### Returns

| Name | Type | Shape | Description |
|---|---|---|---|
| `distances` | `np.ndarray[float32]` | `(n_queries, n_neighbors)` | Squared Euclidean distances to neighbours, sorted ascending. |
| `indices` | `np.ndarray[intp]` | `(n_queries, n_neighbors)` | Indices of neighbours in the corpus (0..n_corpus-1). |

### Errors

Raises `RuntimeError` if Metal is unavailable, shader compilation fails, or
input parameters are invalid (e.g. data length mismatch).

---

## `MetalKNeighbors` (sklearn-style class)

```python
knn = MetalKNeighbors(n_neighbors=5)
knn.fit(data, n, d)
knn.kneighbors(queries, nq)
```

### Constructor

| Argument | Type | Default | Description |
|---|---|---|---|
| `n_neighbors` | `int` | `5` | Number of neighbours to retrieve. |

### Methods

#### `fit(data, n, d) -> MetalKNeighbors`

Store the corpus (database) of points on the GPU. Returns `self` for chaining.

| Param | Type | Description |
|---|---|---|
| `data` | `np.ndarray[float32]` or `list[float]` | Flat row-major corpus points. |
| `n` | `int` | Number of corpus points. |
| `d` | `int` | Number of dimensions. |

The corpus and its pre-computed squared norms are cached on the GPU for
subsequent `kneighbors` calls.

#### `kneighbors(queries, nq) -> Tuple[np.ndarray, np.ndarray]`

Find the `n_neighbors` nearest neighbours of each query point.

| Param | Type | Description |
|---|---|---|
| `queries` | `np.ndarray[float32]` or `list[float]` | Flat row-major query points. |
| `nq` | `int` | Number of query points. |

Returns `(distances, indices)` — see the functional API above for shapes.

**Buffer reuse**: query, norms, and output buffers are cached across calls.
Only a lightweight CPU→GPU copy of query data occurs on each invocation.

### Example

```python
from metal_kmeans import MetalKNeighbors
import numpy as np

corpus = np.random.randn(5000, 32).astype(np.float32)
queries = np.random.randn(100, 32).astype(np.float32)

knn = MetalKNeighbors(n_neighbors=10)
knn.fit(corpus, *corpus.shape)
distances, indices = knn.kneighbors(queries, *queries.shape)
print(distances.shape)  # (100, 10)
print(indices[0])       # indices of 10 nearest neighbours of query 0
```

---

## KNN kernel dispatch

| Condition | Kernel | Description |
|---|---|---|
| D < 32, K ≤ 64 | `knn_assign_dense` | Direct device reads, register-resident query, per-thread heap |
| D ≥ 8, D % 8 == 0, K ≤ 64 | `knn_assign_splitm` | Simdgroup matmul (BN=16, BM=8), shared memory tiling |
| Otherwise | `knn_assign_naive` | Single-thread fallback |

No M-split is used (counterproductive on Apple GPUs due to threadgroup dispatch
overhead). Each threadgroup processes the entire corpus.

All kernels compute a shift-invariant score (`c·c − 2·q·c`); the true squared-L2
distance is recovered by adding `q·q` in the CPU post-process step — this avoids
loading query norms in the inner loop.

## KNN performance notes

- **Buffer reuse**: the first call allocates GPU scratch buffers; subsequent calls
  reuse them (only a CPU→GPU copy of query data occurs). This eliminates the
  per-call `new_buffer` overhead that was the dominant bottleneck.
- **Cold start**: first kernel compile ~20 ms (cached by `fit`).
- **Kernel execution**: typically 7–70 ms depending on shape (see README benchmarks).
- **True L2 distances**: the returned distances are always exact squared-Euclidean,
  not the shift-invariant intermediate scores.

---

## PCA API

Import:

```python
from metal_pca import MetalPCA
```

### `MetalPCA`

sklearn-style PCA with Metal GPU acceleration.

#### Constructor

```python
pca = MetalPCA(n_components)
```

| Argument | Type | Description |
|---|---|---|
| `n_components` | `int` | Number of principal components to compute. |

#### `fit(X, y=None) -> MetalPCA`

Fit PCA to data. `y` is ignored (sklearn Pipeline compatibility).

| Param | Type | Description |
|---|---|---|
| `X` | `np.ndarray[float32]` or `list[float]` | Training data, shape `(n_samples, n_features)`. Supports 2D array or flat list in row-major order. |

**Internal flow**:
1. GPU: mean → center → transpose → matmul (single command buffer).
2. CPU: eigendecomposition (Jacobi ≤ 128, Accelerate LAPACK `ssyevd_` > 128).
3. Sort eigenvalues descending, extract top-K components.

#### `transform(X) -> np.ndarray`

Project data onto principal components (GPU).

| Param | Type | Description |
|---|---|---|
| `X` | `np.ndarray[float32]` or `list[float]` | Data to transform, shape `(n_samples, n_features)`. |

Returns `ndarray[float32]` of shape `(n_samples, n_components)`.

#### `fit_transform(X, y=None) -> np.ndarray`

Fit + transform in one call. Accepts `y` for sklearn Pipeline compatibility.

#### Properties

| Property | Type | Shape | Description |
|---|---|---|---|
| `components_` | `ndarray[float32]` | `(n_components, n_features)` | Principal component vectors (rows). |
| `explained_variance_` | `ndarray[float32]` | `(n_components,)` | Variance explained by each component. |
| `explained_variance_ratio_` | `ndarray[float32]` | `(n_components,)` | Normalized variance (sums to ≤ 1). |
| `singular_values_` | `ndarray[float32]` | `(n_components,)` | Singular values `sqrt(N · λ)`. |
| `mean_` | `ndarray[float32]` | `(n_features,)` | Per-feature mean. |
| `n_features_in_` | `int` | — | Number of features from last fit. |

#### Example

```python
from metal_pca import MetalPCA
import numpy as np

X = np.random.randn(1000, 50).astype(np.float32)

pca = MetalPCA(n_components=5)
pca.fit(X)

print(pca.explained_variance_ratio_)  # [0.42, 0.18, 0.09, ...]
print(pca.components_.shape)          # (5, 50)

transformed = pca.transform(X)        # (1000, 5)
```

#### Performance notes

- **Cold start**: first call compiles 6 Metal shaders (~120 ms total). Subsequent calls reuse cached pipelines.
- **GPU pipeline** is only used for the Gram/covariance matrix computation. Eigendecomposition is always on CPU (Jacobi or Accelerate LAPACK).
- Small datasets (< 10K samples, < 50 features) are faster on CPU — GPU overhead dominates.
- Medium-large shapes (1K-100K samples, 128-8K features) see 1-6× speedup vs CPU.
- sklearn Pipeline compatibility: `make_pipeline(MetalPCA(n_components=k), SomeClassifier())` works with `y=None` in fit methods.
