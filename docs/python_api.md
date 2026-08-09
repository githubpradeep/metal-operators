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

## Gaussian Naive Bayes API

The `metal_gaussian_nb` package provides GPU-accelerated Gaussian Naive Bayes via Apple Metal.

### `metal_gaussian_nb_fit` (functional API)

```python
from metal_gaussian_nb import metal_gaussian_nb_fit

theta, var, prior = metal_gaussian_nb_fit(data, y, n, d, n_classes, var_smoothing=1e-9)
```

- `data` — flat row-major `(n, d)` features (`list[float]` or `np.ndarray[float32]`).
- `y` — class labels `(n,)` as integers in `[0, n_classes)`.
- `n` — number of samples; `d` — number of features; `n_classes` — number of classes (>= 2).
- `var_smoothing` — fraction of the largest per-class variance added to all variances (default `1e-9`).

Returns `(theta_, var_, class_prior_)` — per-class means `(k, d)`, per-class variances `(k, d)`, and empirical priors `(k,)`, all float32.

### `MetalGaussianNB` (sklearn-style class)

```python
from metal_gaussian_nb import MetalGaussianNB

clf = MetalGaussianNB(var_smoothing=1e-9)
clf.fit(data, y, n, d, n_classes)
```

#### Methods

- `fit(data, y, n, d, n_classes) -> MetalGaussianNB` — GPU reduction pass (per-class sum / sum-of-squares), one CPU↔GPU sync.
- `predict(data, n, d) -> np.ndarray[intp]` — argmax class label per sample, shape `(n,)`.
- `predict_log_proba(data, n, d) -> np.ndarray[float32]` — normalized log-probabilities, shape `(n, n_classes)`.
- `predict_proba(data, n, d) -> np.ndarray[float32]` — class probabilities, shape `(n, n_classes)`, rows sum to 1.
- `score(data, y, n, d) -> float` — mean accuracy.

#### Properties

- `theta_` — `(n_classes, d)` per-class feature means (sklearn-style name).
- `var_` — `(n_classes, d)` per-class feature variances after smoothing.
- `class_prior_` — `(n_classes,)` empirical class priors.

#### Performance notes

- Fit = 2 reduction kernels (per-threadgroup partials + deterministic combine) in one command buffer, then a host pass deriving means/variances/priors.
- Predict = 1 kernel computing all per-class log posteriors (`n × k`), with argmax / softmax on the host.
- Shared-memory bound: fit requires `(2·d + 1)·k` floats of threadgroup memory (32 KB cap on Apple Silicon).

## NMF API

Non-negative Matrix Factorization (`V ≈ W·H`, both ≥ 0) via Lee–Seung
multiplicative updates. Every O(N·D·K) step is a Metal matmul.

### Functional API: `metal_nmf_fit`

```python
from metal_operators import metal_nmf_fit

components, coeff, reconstruct_err, n_iter = metal_nmf_fit(
    data, n, d, n_components,
    max_iterations=200, tolerance=1e-4, seed=42
)
```

- `data` — flat row-major, **non-negative** `float32` matrix.
- `n` / `d` — number of rows / columns.
- `n_components` — latent rank `K`.
- Returns fitted `components` (H, K×D), `coeff` (W, N×K), the Frobenius
  reconstruction error `‖V − W·H‖_F`, and the iteration count.
- A `metal_nmf_fit_bytes` variant accepts a `memoryview`/`bytes` buffer
  (raw little-endian float32) for zero-copy numpy inputs.
- Raises `RuntimeError` on negative data values or invalid shapes.

### `MetalNMF` (sklearn-style class)

```python
from metal_operators import MetalNMF

nmf = MetalNMF(n_components=8, max_iterations=200, tolerance=1e-4, seed=42)
nmf.fit(data, n, d)                     # returns None
w_new = nmf.transform(held_out, m, d)   # latent representation of new data
```

#### Methods

- `fit(data, n, d) -> None` — run multiplicative updates.
- `fit_bytes(data, n, d) -> None` — buffer-protocol variant.
- `transform(data, n, d) -> np.ndarray[float32]` — project new data into
  latent space (solves `X ≈ W_new·H` with `H` fixed).
- `transform_bytes(data, n, d) -> np.ndarray[float32]` — buffer variant.

#### Properties

- `components` — `(K, D)` fitted basis (H).
- `coeff` — `(N, K)` coefficient matrix (W) of the training data.
- `reconstruction_error` — `‖V − W·H‖_F`.
- `n_iter` — iterations run by the last `fit`.

## t-SNE API

### Functional API: `metal_tsne`

```python
from metal_operators import metal_tsne

embedding, n_iter, kl = metal_tsne(
    data, n, d,
    n_components=2, perplexity=30.0, learning_rate=200.0,
    n_iter=1000, early_exaggeration=12.0, exaggeration_iter=250,
    momentum=0.8, seed=42, min_grad_norm=1e-7,
)
```

- `data` — flat row-major `float32` matrix of shape `(n, d)`.
- `n_components` — embedding dimension (1..8; 2 is the classic choice).
- `perplexity` — target perplexity; must satisfy `1 <= perplexity < n`.
- Returns the fitted `(n, n_components)` `embedding`, the number of `n_iter`
  iterations actually run, and the final divergences `kl`.
- Early exaggeration (`early_exaggeration` x `P` for `exaggeration_iter`
  iterations) matches scikit-learn. `seed` makes runs reproducible.
- A `metal_tsne_fit_bytes` variant accepts a `memoryview`/`bytes` buffer
  (raw little-endian float32) for zero-copy numpy inputs.
- Raises `RuntimeError` on invalid shapes/parameters.

### `MetalTSNE` (sklearn-style class)

```python
from metal_tsne import MetalTSNE

tsne = MetalTSNE(n_components=2, perplexity=30.0, n_iter=1000, seed=42)
Y = tsne.fit_transform(data)   # (n, 2) embedding as float32 ndarray
print(tsne.kl_divergence_)
```

#### Methods

- `fit(data, n=None, d=None) -> MetalTSNE` — fit; infers `(n, d)` from shape.
- `fit_transform(data, n=None, d=None) -> np.ndarray[float32]` — fit and return
  the embedding.

#### Properties

- `embedding_` — `(n, n_components)` low-dimensional embedding (float32).
- `n_iter_` — iterations run by the last `fit`.
- `kl_divergence_` — final KL(P‖Q) of the embedding.

#### GPU breakdown

The exact, per-iteration squared-L2 distance / affinity gradient kernels
(`tsne_distances`, `tsne_perplexity`, `tsne_grad` in `shaders/tsne.metal`)
offload the two dominant O(N²) costs — building the pairwise affinities and the
per-iteration gradient — onto the Metal GPU. The `O(N)` momentum update stays
on the CPU. This is the largest-lift operator: exact t-SNE otherwise spends
all of its time in these quadratic loops.

---

## GMM API

Gaussian Mixture Model (full covariance, EM, k-means++ init) mirroring
`sklearn.mixture.GaussianMixture`. The per-iteration E-step — the O(n·k·d²)
log-likelihood matrix — runs on the GPU (`gmm_e_step` in `shaders/gmm.metal`);
the M-step and the per-component Cholesky precision/logdet update run on the
host.

### Functional API: `metal_gmm`

```python
from metal_gmm import metal_gmm

weights, means, covariances, responsibilities, lower_bound, n_iter = metal_gmm(
    data, n, d, n_components=3, max_iterations=100, tolerance=1e-3, seed=42,
    reg_covar=1e-6,
)
```

Returns:

- `weights` — `(k,)` mixture weights (sum to 1), float32
- `means` — `(k, d)` component means, float32
- `covariances` — `(k, d, d)` full covariance matrices, float32
- `responsibilities` — `(n, k)` posteriors of the fitted data, float32
- `lower_bound` — final average log-likelihood (float)
- `n_iter` — EM iterations run (int)

A `metal_gmm_fit_bytes` variant accepts a `memoryview`/`bytes` buffer instead
of a list.

### `MetalGMM` (sklearn-style class)

```python
from metal_gmm import MetalGMM

gmm = MetalGMM(n_components=3, max_iterations=100, tolerance=1e-3, seed=42)
gmm.fit(data, n=None, d=None)        # infers (n, d) from the array shape
labels = gmm.predict(new_data)       # (n,) hard component assignment (intp)
proba = gmm.predict_proba(new_data)  # (n, k) responsibilities, rows sum to 1
ll = gmm.score(new_data)             # average log-likelihood
```

#### Methods

- `fit(data, n=None, d=None) -> MetalGMM` — fit by EM; infers `(n, d)` from
  shape.
- `fit_predict(data, n=None, d=None) -> np.ndarray[intp]` — fit and return the
  hard assignment.
- `predict(data, n=None, d=None) -> np.ndarray[intp]` — most likely component.
- `predict_proba(data, n=None, d=None) -> np.ndarray[float32]` — posterior
  component probabilities.
- `score(data, n=None, d=None) -> float` — average log-likelihood.

#### Properties

- `weights_` — `(k,)` fitted mixture weights.
- `means_` — `(k, d)` fitted component means.
- `covariances_` — `(k, d, d)` fitted full covariances.
- `responsibilities_` — `(n, k)` posteriors of the training data.
- `lower_bound_` — final average log-likelihood lower bound.
- `n_iter_` — EM iterations run by the last `fit`.

#### GPU breakdown

`predict` / `predict_proba` / `score` are each a single GPU launch; `fit` runs
one GPU launch per EM iteration (the E-step) with the M-step on the host —
typically 2–30 iterations for well-behaved problems.

## SVC API

The `metal_svm` package provides GPU-accelerated Support Vector Classification
(`sklearn.svm.SVC`-style). The full `n×n` kernel (Gram) matrix and the decision
matrix are computed on the GPU (`svm_kernel` / `svm_predict` in
`shaders/svm.metal`); the per-iteration dual (α) updates run on the host as a
simplified Platt SMO that reads the precomputed Gram — so the GPU does the
O(n²·d) kernel work exactly once, shared by every one-vs-rest classifier.

### Functional API: `metal_svc`

```python
from metal_svm import metal_svc

classes, intercept_, dual_coef_, support_vectors_, support_count, gamma_, n_iter = (
    metal_svc(X, y, n, d, kernel="rbf", gamma=0.5, c=10.0,
              max_iter=200, tolerance=1e-4, seed=42)
)
```

Returns:

- `classes` — `(k,)` unique class labels, float32
- `intercept_` — `(k,)` per-classifier intercept `b`, float32
- `dual_coef_` — `(ns,)` pooled `α·y` duals, float32
- `support_vectors_` — `(ns, d)` pooled support vectors, float32
- `support_count` — total support vectors across all classifiers (int)
- `gamma_` — resolved kernel width (float)
- `n_iter` — `(k,)` SMO passes per classifier

A `metal_svc_fit_bytes` variant accepts `memoryview`/`bytes` buffers. The
functional API does not retain the fitted model; use `MetalSVC` for
`predict` / `decision_function` / `score`.

### `MetalSVC` (sklearn-style class)

```python
from metal_svm import MetalSVC

clf = MetalSVC(kernel="rbf", gamma=0.5, c=10.0, degree=3.0, coef0=0.0,
               tolerance=1e-3, max_iter=200, seed=42)
clf.fit(X, y)                  # infers (n, d) from the array shape
preds = clf.predict(new_X)     # (n,) hard labels
dec = clf.decision_function(new_X)  # (n, k) raw scores (or (n,) for binary)
acc = clf.score(X_test, y_test)
```

`gamma <= 0` selects the automatic default `1 / n_features` (the
`1/n_features` fallback of sklearn's `gamma="scale"`). Kernels: `"linear"`,
`"poly"` (uses `degree`), `"rbf"` (default), `"sigmoid"` (uses `coef0`).

#### Methods

- `fit(data, y, n=None, d=None) -> MetalSVC` — fit with the GPU Gram matrix
  + host SMO; infers `(n, d)` from shape.
- `predict(data, n=None, d=None) -> np.ndarray[intp]` — hard labels (sign for
  binary, argmax over the one-vs-rest scores otherwise).
- `decision_function(data, n=None, d=None) -> np.ndarray[float32]` — raw
  scores `(n, k)`; the single column for a binary classifier.
- `score(data, y, n=None, d=None) -> float` — mean accuracy.

#### Properties

- `classes_` — `(k,)` unique class labels.
- `intercept_` — `(k,)` per-classifier intercept `b`.
- `n_support_` — `(k,)` support-vector count per classifier.
- `support_vectors_` — `(ns, d)` pooled support vectors.
- `dual_coef_` — `(ns,)` pooled `α·y` dual coefficients.
- `n_iter_` — `(k,)` SMO passes run per classifier.
- `gamma_` — resolved kernel width used by `fit`.

#### GPU breakdown

`fit` runs exactly **one** kernel-matrix launch (the O(n²·d) Gram, reused by
every one-vs-rest sub-problem) plus the host SMO loop; `predict` /
`decision_function` / `score` are each a single `svm_predict` launch over the
pooled support vectors. Note the SMO dual solve itself is host-side (O(n²) per
pass), so this operator shines when the Gram or the decision matrix dominates
— i.e. prediction / scoring on large test sets, and any kernel besides linear.

## SVR API

The `metal_svr` package provides GPU-accelerated Support Vector Regression
(`sklearn.svm.SVR`-style, ε-insensitive loss). It reuses the **exact same two
shaders** as SVC: `svm_kernel` builds the full `n×n` kernel (Gram) matrix in
one GPU launch, and `svm_predict` computes the regression outputs
`f(x) = b + Σᵢ βᵢ·κ(x, xᵢ)` over the pooled support vectors (a single
classifier, so one launch). The host runs an ε-insensitive SMO on the dual
coefficients `βᵢ = αᵢ⁺ - αᵢ⁻ ∈ [-C, C]` with `Σβ = 0` and the ε-tube penalty.

### Functional API: `metal_svr`

```python
from metal_svr import metal_svr

support_vectors_, dual_coef_, intercept_, support_count, gamma_, n_iter = (
    metal_svr(X, y, n, d, kernel="rbf", gamma=0.6, c=5.0, eps=0.05,
              max_iter=300, tolerance=1e-3, seed=42)
)
```

Returns:

- `support_vectors_` — `(ns, d)` pooled support vectors, float32
- `dual_coef_` — `(ns,)` dual coefficients `β = α⁺ - α⁻`, float32
- `intercept_` — regression intercept `b` (float)
- `support_count` — number of support vectors (int)
- `gamma_` — resolved kernel width (float)
- `n_iter` — SMO passes run (int)

A `metal_svr_fit_bytes` variant accepts `memoryview`/`bytes` buffers. The
functional API does not retain the fitted model; use `MetalSVR` for
`predict` / `decision_function` / `score`.

### `MetalSVR` (sklearn-style class)

```python
from metal_svr import MetalSVR

clf = MetalSVR(kernel="rbf", gamma=0.6, c=5.0, eps=0.05, degree=3.0, coef0=0.0,
               tolerance=1e-3, max_iter=200, seed=42)
clf.fit(X, y)                  # infers (n, d) from the array shape
preds = clf.predict(new_X)     # (n,) regression values
dec = clf.decision_function(new_X)  # identical to predict (single regressor)
r2 = clf.score(X_test, y_test)
```

`gamma <= 0` selects the automatic default `1 / n_features` (the
`1/n_features` fallback of sklearn's `gamma="scale"`). Kernels: `"linear"`,
`"poly"` (uses `degree`), `"rbf"` (default), `"sigmoid"` (uses `coef0`).

#### Methods

- `fit(data, y, n=None, d=None) -> MetalSVR` — fit with the GPU Gram matrix
  + host ε-SMO; infers `(n, d)` from shape.
- `predict(data, n=None, d=None) -> np.ndarray[float32]` — regression values
  `(n,)`.
- `decision_function(data, n=None, d=None) -> np.ndarray[float32]` — same as
  `predict` (SVR is a single real-valued regressor).
- `score(data, y, n=None, d=None) -> float` — R² coefficient of
  determination, `1 - SS_res / SS_tot` (clipped to 0).

#### Properties

- `support_vectors_` — `(ns, d)` pooled support vectors.
- `dual_coef_` — `(ns,)` dual coefficients `β = α⁺ - α⁻`.
- `intercept_` — regression intercept `b` (float).
- `n_support_` — number of support vectors (int).
- `n_iter_` — SMO passes run (int).
- `gamma_` — resolved kernel width used by `fit`.

#### GPU breakdown

`fit` runs exactly **one** kernel-matrix launch (the O(n²·d) Gram — shared
with SVC, which is why `metal_svm` and `metal_svr` compile the same two
shaders) plus the host ε-SMO loop; `predict` / `decision_function` / `score`
are each a single `svm_predict` launch over the pooled support vectors. As
with SVC, the dual solve is host-side (O(n²) per pass), so prediction /
scoring on large test sets is where the GPU work dominates.

## Lasso API

`metal_lasso` (functional) and `MetalLasso` (sklearn-style class), mirroring
`sklearn.linear_model.Lasso`. Minimizes `0.5·‖Xw − y‖² + alpha·‖w‖₁`.

### `metal_lasso(data, y, n, d, alpha=1.0, fit_intercept=True, max_iterations=1000, tol=1e-4, seed=42)`

Returns `(weights, intercept, n_iter, converged, final_loss)`:

- `weights` — `(d,)` float32 coefficient vector.
- `intercept` — float intercept (0.0 when `fit_intercept=False`).
- `n_iter` — coordinate-descent sweeps run.
- `converged` — `bool`, whether the solver converged within `tol`.
- `final_loss` — mean squared error on the training data.

### `MetalLasso(alpha=1.0, fit_intercept=True, max_iterations=1000, tol=1e-4, seed=42)`

sklearn-style class with `fit(data, y, n, d)`, `predict(data, n, d)`,
`score(data, y, n, d)`, and properties `coef_`, `intercept_`, `weights`,
`bias`, `n_iter`, `converged`, `final_loss`.

Alias underscores mirror sklearn names (`coef_`, `intercept_`).

#### GPU breakdown

`fit` builds the augmented `(d+1)²` Gram system once on the GPU (the shared
`linreg_gram_*` kernels), then runs host coordinate descent over the cached
matrix — each sweep is O(d²), independent of the sample count `n`. Once the
Gram is built, tuning `alpha` never re-reads the dataset. `predict` / `score`
are each a single shared `linreg_predict` launch.
