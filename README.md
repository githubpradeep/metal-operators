# metal-operators

GPU-accelerated **KMeans clustering**, **K-Nearest Neighbors**, **PCA**, **LDA**, **Logistic Regression**, **Linear Regression**, **Gaussian Naive Bayes**, **Gaussian Mixture Models (GMM)**, and **Support Vector Classification (SVC)** via Apple Metal.

**KMeans** uses 5 kernel variants (simdgroup, split-D, tiled centroid) to run Lloyd's
algorithm entirely on GPU — no CPU readback inside the loop.

**KNN** uses 3 kernel variants (Dense direct-read for D<32, simdgroup Splitm for
D≥32&D%8=0, Naive fallback) with buffer reuse for 2.6–22× speedup over BLAS-CPU.

**LogisticRegression** uses 3 fused forward/backward kernel variants (simdgroup,
split-D, naive) plus a dedicated predict kernel for full-batch L-BFGS (m = 10)
training: one GPU launch evaluates loss + gradient over the whole dataset.

**LinearRegression** solves the normal equations in closed form: a tiled
shared-memory Gram kernel (D ≤ 128) and a chunked Xᵀy/column-sum kernel build
the augmented system, a deterministic reduction combines per-group partials,
and the (D+1)² system is solved on the host — matching sklearn's `cholesky`
solver with optional L2 ridge.

**LDA** is a supervised linear dimensionality-reduction: a single batched
GPU scatter (Gram) kernel computes `Xᵀ·X`, the within/between-class scatter
matrices are assembled from it on the host, and the small symmetric
generalized eigenproblem `S_w⁻¹·S_b` is solved via whitening (Accelerate
`ssyevd`) to yield the top-k discriminant axes + a nearest-class-center
classifier.

```python
from metal_kmeans import metal_kmeans, MetalKMeans, metal_kneighbors, MetalKNeighbors
import numpy as np

# ── KMeans ──────────────────────────────────────────────────────
data = np.random.randn(10000, 32).astype(np.float32)
labels, centroids, n_iter, inertia = metal_kmeans(
    data.ravel().tolist(), *data.shape, n_clusters=8,
    max_iterations=20, tolerance=1e-4, seed=42,
)

# ── KNN ─────────────────────────────────────────────────────────
corpus = np.random.randn(5000, 32).astype(np.float32)
queries = np.random.randn(100, 32).astype(np.float32)

# Functional API
distances, indices = metal_kneighbors(
    corpus.ravel().tolist(), *corpus.shape,
    queries.ravel().tolist(), *queries.shape,
    n_neighbors=5,
)

# sklearn-style API
knn = MetalKNeighbors(n_neighbors=5)
knn.fit(corpus, *corpus.shape)
distances, indices = knn.kneighbors(queries, *queries.shape)
```

## Install

```sh
pip install maturin
git clone https://github.com/YOUR_USER/metal-operators
cd metal-operators
maturin develop
```

Metal shaders compile on first call (~20 ms/kernel); subsequent calls reuse cached pipelines.

## Examples

```sh
python3 examples/example.py                # KMeans: smoke test + benchmark
python3 examples/knn_example.py            # KNN: benchmark across shapes
python3 examples/movie_recommendation.py   # KNN: movie recommendation engine
python3 examples/customer_segmentation.py  # KMeans: 500K customer segmentation
python3 examples/pca_eigenfaces.py         # PCA: eigenfaces reconstruction
python3 examples/logistic_regression_example.py  # LogisticRegression: smoke test + benchmark
python3 examples/linear_regression_example.py    # LinearRegression: smoke test + benchmark
python3 examples/diabetes_regression.py          # LinearRegression: real diabetes data (442×10)
python3 examples/lda_example.py                  # LDA: supervised dimensionality reduction
python3 examples/tsne_example.py                 # t-SNE: nonlinear embedding (largest-lift)
python3 examples/gmm_example.py                  # GMM: Gaussian mixture model (EM)
python3 examples/svm_example.py                  # SVC: kernel matrix + host SMO classifier
```

### PCA

```python
from metal_pca import MetalPCA
import numpy as np

# Synthetic data with known low-rank structure
n, d, k = 1000, 50, 5
rng = np.random.RandomState(42)
X = rng.randn(n, k) @ rng.randn(k, d) + 0.1 * rng.randn(n, d)
X = X.astype(np.float32)

# sklearn-style API
pca = MetalPCA(n_components=k)
pca.fit(X)

# Access results
components = pca.components_           # (k, d) principal components
ev = pca.explained_variance_           # (k,) eigenvalues
ev_ratio = pca.explained_variance_ratio_  # (k,) normalized
transformed = pca.transform(X)         # (n, k) projection

# Reconstruction
X_recon = transformed @ components
recon_error = np.mean((X - X_recon)**2)
```

### LDA

Supervised linear dimensionality reduction (sklearn-style
`LinearDiscriminantAnalysis`). A batched GPU scatter kernel computes the Gram
matrix `Xᵀ·X`; the within-class (`S_w`) and between-class (`S_b`) scatter
matrices are assembled on the host, and the top-k axes of the generalized
eigenproblem `S_w⁻¹·S_b` are found via whitening (Accelerate `ssyevd`).
`predict`/`score` classify by nearest projected class center.

```python
from metal_lda import MetalLDA
import numpy as np

# 3 Gaussian classes, 12 features, 2 informative directions
rng = np.random.RandomState(42)
n, d, k = 600, 12, 2
X = rng.randn(n, d).astype(np.float32)
y = np.concatenate([np.full(200, c, np.float32) for c in (0, 1, 2)])
X[y == 1, 0] += 4.0
X[y == 2, 1] += 4.0

# sklearn-style API
lda = MetalLDA(n_components=k)
lda.fit(X, y)

scalings = lda.scalings_              # (k, d) discriminant axes
evals = lda.eigenvalues_              # (k,) discriminative power (descending)
transformed = lda.transform(X)        # (n, k) projection
preds = lda.predict(X)                # class indices 0..C-1
acc = lda.score(X, y)                 # accuracy
```

### Logistic Regression

Binary logistic regression trained with full-batch L-BFGS (m = 10). A fused
forward/backward/loss Metal kernel (simdgroup for D≥8&D%8==0, split-D for
D>128, naive fallback) evaluates the whole dataset in one launch per
iteration; per-threadgroup partials are combined by a tiny deterministic
reduction kernel in the same command buffer — one CPU↔GPU sync per
iteration (no per-batch readback, no device atomics), plus a dedicated
predict kernel. The two-stage reduction makes loss/gradient bitwise
deterministic across runs.

```python
from metal_logistic_regression import metal_logistic_regression, MetalLogisticRegression
import numpy as np

# Synthetic two-blob binary classification data
rng = np.random.RandomState(42)
n, d = 2000, 8
X = np.vstack([rng.randn(n // 2, d) - 1.5, rng.randn(n // 2, d) + 1.5]).astype(np.float32)
y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float32)

# Functional API — returns (weights, bias, n_epochs, final_loss)
weights, bias, n_epochs, loss = metal_logistic_regression(
    X.ravel().tolist(), y.tolist(), n, d,
    c=1.0, learning_rate=0.02, max_epochs=50, seed=42,
)

# sklearn-style API
clf = MetalLogisticRegression(c=1.0, learning_rate=0.02, max_epochs=50, seed=42)
clf.fit(X, y, n, d)

# Access results
coef = clf.coef_                 # (d,) learned coefficients
intercept = clf.intercept_       # scalar bias
proba = clf.predict_proba(X, n, d)  # (n,) probabilities
preds = clf.predict(X, n, d)         # (n,) hard labels {0, 1}
acc = clf.score(X, y, n, d)          # mean accuracy
```

Labels must be `{0.0, 1.0}`; `C` follows sklearn convention (inverse
regularization strength). Training stops early when the gradient sup-norm
drops below `tol`. `learning_rate`, `momentum`, `batch_size` and `seed` are
accepted for API symmetry with flashlib but are not used by the optimizer.

### Linear Regression

Linear regression solved exactly in closed form via the normal equations
(matching sklearn's `LinearRegression`/`Ridge` with the default `cholesky`
solver). Two streaming Metal kernels build the augmented Gram system
(`[XᵀX | Xᵀ·1; 1ᵀ·X | n]` and `[Xᵀy; Σy]`): the XᵀX kernel stages row blocks
in shared memory for D ≤ 128 (tiled; ideal `n·d` memory traffic) or uses one
thread per element for D > 128, the Xᵀy kernel computes column sums /
Xᵀy / Σy from chunked element columns, and a tiny deterministic reduction
combines the per-group partials — no device atomics, two CPU↔GPU syncs
total. The (D+1)×(D+1) system is solved on the host with Gaussian
elimination + partial pivoting, and a dedicated predict kernel evaluates the
model.

```python
from metal_linear_regression import metal_linear_regression, MetalLinearRegression
import numpy as np

# Synthetic data with known coefficients
rng = np.random.RandomState(42)
n, d = 2000, 8
X = (rng.randn(n, d) * 2.0 - 1.0).astype(np.float32)
w_true = rng.randn(d).astype(np.float32)
y = (X @ w_true + 1.5 + 0.1 * rng.randn(n)).astype(np.float32)

# Functional API — returns (weights, bias, n_iter, final_loss)
weights, bias, n_iter, mse = metal_linear_regression(
    X.ravel().tolist(), y.tolist(), n, d, alpha=0.0, fit_intercept=True,
)

# sklearn-style API
reg = MetalLinearRegression(alpha=0.0, fit_intercept=True)
reg.fit(X, y, n, d)

# Access results
coef = reg.coef_                 # (d,) learned coefficients
intercept = reg.intercept_       # scalar bias
preds = reg.predict(X, n, d)     # (n,) predictions
r2 = reg.score(X, y, n, d)       # coefficient of determination (R²)
```

`alpha > 0` adds L2 ridge regularization on the coefficients (sklearn
`Ridge` convention); `fit_intercept=False` drops the bias column from the
augmented system. `max_iterations`, `tol` and `seed` are accepted for API
symmetry with flashlib but are unused by the closed-form solver.

### Gaussian Naive Bayes

Gaussian Naive Bayes (multiclass, sklearn `GaussianNB` conventions): a GPU
reduction pass computes the per-class feature sum and sum-of-squares in one
CPU↔GPU sync (two kernels — per-threadgroup partials, then a deterministic
fixed-order combine — no device atomics), after which the host derives
per-class means, variances (with `var_smoothing`), and empirical priors. A
dedicated predict kernel computes per-class log posteriors; the host applies
the argmax (`predict`) and the log-sum-exp softmax (`predict_log_proba` /
`predict_proba`).

```python
from metal_gaussian_nb import metal_gaussian_nb_fit, MetalGaussianNB
import numpy as np

# Synthetic 3-class data (separable means)
rng = np.random.RandomState(0)
n, d, k = 300, 4, 3
X = rng.randn(n, d).astype(np.float32)
y = (rng.randint(0, k, n)).astype(np.float32)
X += y[:, None] * 2.0  # shift classes apart

# sklearn-style API — fit needs n_classes
clf = MetalGaussianNB()
clf.fit(X.ravel().tolist(), y.tolist(), n, d, n_classes=k)

# Access results
theta = clf.theta_         # (k, d) per-class means
var = clf.var_             # (k, d) per-class variances
prior = clf.class_prior_   # (k,) empirical priors
proba = clf.predict_proba(X, n, d)  # (n, k) class probabilities
preds = clf.predict(X, n, d)        # (n,) predicted labels
acc = clf.score(X, y, n, d)         # mean accuracy
```

### Gaussian Mixture Model (GMM)

Gaussian Mixture Model (full covariance, sklearn `GaussianMixture` semantics)
fitted by Expectation-Maximization. k-means++ picks the seed means; each EM
iteration's E-step — the O(n·k·d²) log-likelihood matrix — runs on the GPU
(`gmm_e_step` in `shaders/gmm.metal`, one launch per iteration), while the host
computes responsibilities, the M-step (weighted mean/covariance update), and
the per-component Cholesky factorization that rebuilds `Σ⁻¹` and `log|Σ|`.
`predict` / `predict_proba` / `score` are each a single GPU launch.

```python
from metal_gmm import MetalGMM, metal_gmm
import numpy as np

# Three well-separated blobs
rng = np.random.RandomState(7)
X = np.concatenate([
    rng.randn(200, 3) + np.array([6, 0, 0]),
    rng.randn(200, 3) + np.array([0, 6, 0]),
    rng.randn(200, 3) + np.array([0, 0, 6]),
]).astype(np.float32)

# sklearn-style API
gmm = MetalGMM(n_components=3, seed=42)
gmm.fit(X)
means = gmm.means_          # (3, 3) component means
proba = gmm.predict_proba(X)  # (n, 3) responsibilities, rows sum to 1
labels = gmm.predict(X)     # (n,) hard component assignment
ll = gmm.score(X)           # average log-likelihood

# Functional API — (weights, means, covariances, responsibilities, lb, n_iter)
w, means, covs, resp, lb, iters = metal_gmm(X, *X.shape, n_components=3)
```

### Support Vector Classification (SVC)

`sklearn.svm.SVC`-semantics classifier (one-vs-rest, RBF by default). The full
`n×n` kernel (Gram) matrix — the O(n²·d) term — is built in a single `svm_kernel`
GPU launch and, being a function of the data alone, is reused verbatim by every
one-vs-rest binary sub-problem. The per-iteration dual (α) updates run on the
host as a simplified Platt SMO reading that Gram; `predict` / `decision_function`
/ `score` are each a single `svm_predict` launch over the pooled support vectors.

```python
from metal_svm import MetalSVC
import numpy as np

# XOR (not linearly separable) — RBF kernel
X = np.array([[0,0],[1,1],[0,1],[1,0]]).astype(np.float32)
y = np.array([0,0,1,1]).astype(np.float32)

# sklearn-style API
clf = MetalSVC(kernel="rbf", gamma=0.5, c=100.0)
clf.fit(X, y)
preds = clf.predict(X)          # (n,) hard labels
dec = clf.decision_function(X)  # (n, n_classes) raw scores
acc = clf.score(X, y)

# Functional API — (classes, intercept_, dual_coef_, support_vectors_,
# support_count, gamma_, n_iter)
from metal_svm import metal_svc
classes, intercept, dual, sv, ns, g, iters = metal_svc(
    X, y, *X.shape, kernel="rbf", gamma=0.5, c=100.0
)
```

## Requirements

- macOS (Apple Silicon or AMD GPU with Metal support)
- Python 3.9+
- Rust toolchain (one-time `maturin develop` only)

No Xcode required — Metal shaders compile at runtime from inline source.

## API

### Functional

```python
metal_kmeans(data, n, d, n_clusters, max_iterations=100,
             tolerance=1e-4, seed=42) -> (labels, centroids, n_iter, inertia)
```

| Returns | Type | Shape |
|---|---|---|
| `labels` | `np.ndarray[intp]` | `(n,)` |
| `centroids` | `np.ndarray[float32]` | `(n_clusters, d)` |
| `n_iter` | `int` | — |
| `inertia` | `float` | — |

### sklearn-style

```python
km = MetalKMeans(n_clusters=8, max_iterations=20, tolerance=1e-4, seed=42)
km.fit(data, n, d)          # data: list[float] or np.ndarray[float32]
km.predict(new_data, n, d)  # assign to fitted centroids
km.cluster_centers_         # (n_clusters, d) float32
km.labels_                  # per-point labels from last fit
km.inertia_                 # within-cluster sum of squared distances
km.n_iter_                  # iterations used
```

`data` is flat row-major: `data[i * d + j]` = point `i`, dimension `j`.

## KNN API

### Functional

```python
distances, indices = metal_kneighbors(
    corpus, n_corpus, d, queries, n_queries, n_neighbors=5
)
```

| Returns | Type | Shape |
|---|---|---|
| `distances` | `np.ndarray[float32]` | `(n_queries, n_neighbors)` |
| `indices` | `np.ndarray[intp]` | `(n_queries, n_neighbors)` |

Distances are **squared Euclidean** (true L2, not shift-invariant).

### sklearn-style

```python
knn = MetalKNeighbors(n_neighbors=5)
knn.fit(data, n, d)                     # corpus (database) of points
dist, idx = knn.kneighbors(queries, nq)  # find nearest neighbours
```

### Kernel dispatch

| Condition | Kernel | Description |
|---|---|---|
| D < 32, K ≤ 64 | `knn_assign_dense` | Direct device reads, query register-resident, per-thread heap |
| D ≥ 8, D % 8 == 0, K ≤ 64 | `knn_assign_splitm` | Simdgroup matmul (BN=16, BM=8), shared memory, M-split disabled |
| Otherwise | `knn_assign_naive` | Single-thread fallback, each threadgroup processes one query |

No M-split is used on Apple GPUs (threadgroup dispatch overhead ~50 µs makes it
counterproductive). Each threadgroup processes the entire corpus.

All three kernels compute a shift-invariant score (`c·c − 2·q·c`); the true
squared-L2 distance is recovered by adding `q·q` in the CPU post-process step
(avoids loading query norms in the inner loop).

### KNN benchmarks (Apple M3, buffer reuse enabled)

| Queries | Corpus | D | K | Metal | BLAS-CPU | Speedup |
|---|---|---|---|---|---|---|
| 1K | 10K | 8 | 5 | **13.5 ms** | 35 ms | **2.6×** |
| 1K | 10K | 32 | 5 | **7.3 ms** | 35 ms | **4.8×** |
| 1K | 50K | 8 | 5 | **62.5 ms** | 167 ms | **2.7×** |
| 1K | 50K | 32 | 5 | **34.9 ms** | 171 ms | **4.9×** |
| 10K | 10K | 8 | 5 | **15.4 ms** | 331 ms | **21.5×** |
| 10K | 10K | 32 | 5 | **39.3 ms** | 338 ms | **8.6×** |
| 10K | 50K | 8 | 5 | **71.8 ms** | 1.60 s | **22.3×** |

Larger problems show the biggest gains: the GPU's parallelism and buffer reuse
overwhelm the CPU's cache-limited BLAS path.

## Rust API (advanced)

For direct Rust integration:

```rust
use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

let ctx = MetalContext::new()?;
let mut km = KMeans::new(KMeansConfig { k: 8, max_iterations: 15, .. });
km.fit(&ctx, &data, n, d)?;
println!("inertia: {}", km.inertia());
```

```rust
use metal_operators::knn::{KNN, KNNConfig};

let ctx = MetalContext::new()?;
let mut knn = KNN::new(KNNConfig { k: 5 });
knn.fit(&ctx, &corpus, nc, d)?;
let (dists, idxs) = knn.kneighbors(&ctx, &queries, nq)?;
```

```rust
use metal_operators::pca::{PCA, PCAConfig};

let ctx = MetalContext::new()?;
let mut pca = PCA::new(PCAConfig { n_components: 5 });
pca.fit(&ctx, &data, n, d)?;
println!("explained variance: {:?}", &pca.explained_variance());
let transformed = pca.transform(&ctx, &data, n, d)?;
```

```rust
use metal_operators::logistic_regression::{LogisticRegression, LogisticRegressionConfig};

let ctx = MetalContext::new()?;
let mut lr = LogisticRegression::new(LogisticRegressionConfig {
    c: 1.0, learning_rate: 0.02, max_epochs: 50, ..Default::default()
});
lr.fit(&ctx, &data, &y, n, d)?;
println!("loss: {}", lr.final_loss);
let probs = lr.predict_proba(&ctx, &data, n, d)?;
```

```rust
use metal_operators::linear_regression::{LinearRegression, LinearRegressionConfig};

let ctx = MetalContext::new()?;
let mut lr = LinearRegression::new(LinearRegressionConfig {
    alpha: 0.0, fit_intercept: true, ..Default::default()
});
lr.fit(&ctx, &data, &y, n, d)?;
println!("weights: {:?}", lr.weights());
println!("bias:    {}", lr.intercept());
let preds = lr.predict(&ctx, &data, n, d)?;
println!("R²:      {}", lr.score(&ctx, &data, &y, n, d)?);
```

```rust
use metal_operators::naive_bayes::{GaussianNB, GaussianNBConfig};

let ctx = MetalContext::new()?;
// `y` holds class ids in [0, n_classes); k = 3 classes here.
let mut nb = GaussianNB::new(GaussianNBConfig { var_smoothing: 1e-9 });
nb.fit(&ctx, &data, &y, n, d, 3)?;
println!("means: {:?}", nb.means());
let preds = nb.predict(&ctx, &data, n, d)?;          // argmax class per sample
let proba = nb.predict_proba(&ctx, &data, n, d)?;    // (n, k) softmax rows
let acc = nb.score(&ctx, &data, &y, n, d)?;          // mean accuracy
```

```rust
use metal_operators::svm::{SVC, SVCConfig, SVCKernel};

let ctx = MetalContext::new()?;
// XOR-like 2-D pattern: separable in RBF space, not linearly.
let mut svc = SVC::new(SVCConfig {
    kernel: SVCKernel::Rbf, gamma: 0.5, c: 100.0,
    tolerance: 1e-6, max_iter: 400, seed: 3, ..Default::default()
});
svc.fit(&ctx, &data, &labels, 4, 2)?;
let preds = svc.predict(&ctx, &data, 4, 2)?;   // one-vs-rest argmax / sign
let dec = svc.decision_function(&ctx, &data, 4, 2)?; // raw (4, 2) scores
```

## Tests

```sh
cargo test                     # Rust: 30+ integration tests
python3 examples/example.py    # Python KMeans smoke test
python3 examples/knn_example.py # Python KNN smoke test
python3 examples/pca_eigenfaces.py # Python PCA eigenfaces example
python3 examples/logistic_regression_example.py # Python LogisticRegression smoke test
python3 examples/linear_regression_example.py   # Python LinearRegression smoke test
python3 examples/svm_example.py                 # Python SVC smoke test
```

KMeans test matrix: D = {2, 4, 8, 16, 32, 64, 128}, K = {1, 8, 16, 32, 33, 64, 256}, including adjusted Rand index validation against CPU reference, multi-simdgroup correctness, split-D, empty-cluster handling, and timing.

KNN test matrix: D = {3, 8, 16, 32}, K = {1, 3, 5, 10}, covering Dense, Splitm, and Naive kernel paths plus deterministic reproducibility.

PCA test matrix: 12 tests covering cov path (N≥D), Gram path (N<D), explained variance ordering, orthonormal components, reconstruction accuracy, single-component edge case, and transform output shape.

Linear regression test matrix: 10 tests covering recovery of known coefficients, R² accuracy vs CPU reference, ridge shrinkage, no-intercept mode, singular data, determinism, and kernel dispatch sweep (D = {2, 3, 8, 16, 64, 128, 256}).

Gaussian NB test matrix: CPU-reference validation of per-class means/variances from the GPU reduction pass (single-group, multi-group >128 samples, and 3-class), and classification accuracy + probability-normalization checks for the predict kernel.

SVC test matrix: linear-kernel separable-blob accuracy + decision-matrix-argmax agreement, RBF nonlinear (XOR) separation, deterministic seed reproducibility, and invalid-input guards (not-fitted, single class, non-positive C).

## Benchmarks

```sh
cargo bench --bench kmeans_benchmark
cargo bench --bench kmeans_benchmark -- "fit.*1M_D=32"
cargo bench --bench pca_benchmark          # PCA: tall/square/wide regimes
```

Results (Apple M3):

### Assign (single iteration)

| N | D | K | Metal | BLAS-CPU | Speedup |
|---|---|---|---|---|---|
| 1M | 2 | 8 | 0.89 ms | 20.7 ms | **23×** |
| 100K | 2 | 8 | 0.30 ms | 2.04 ms | **6.7×** |
| 10K | 64 | 16 | 1.78 ms | 0.57 ms | 0.3× |
| 10K | 128 | 32 | 6.6 ms | 1.05 ms | 0.16× |

### Fit (15 iterations)

| N | D | K | Metal | BLAS-CPU | sklearn | vs BLAS |
|---|---|---|---|---|---|---|
| 100K | 32 | 256 | 32 ms | 82 ms | 33 ms | **2.6×** |
| 1M | 32 | 64 | 128 ms | 211 ms | 140 ms | **1.65×** |
| 1M | 32 | 16 | 36 ms | 108 ms | 99 ms | **3.0×** |
| 3M | 32 | 16 | 98 ms | 387 ms | — | **3.9×** |
| 200K | 64 | 512 | 306 ms | 348 ms | 118 ms | **1.1×** |
| 50K | 2 | 8 | 1.2 ms | 2.5 ms | 3.7 ms | **2.0×** |

The GPU centroid update (`kmeans_centroid_tiled`) replaces the CPU bottleneck (12 ms → <1 ms warm for 1M×32×16), yielding ~3× fit speedup. 2D shapes show the largest gains from assign (23×). High-D (128D) loses — 8×8 tile size doesn't fill the GPU's matrix units.

### PCA fit benchmarks (Apple M3)

PCA uses: GPU for mean → center → transpose → matmul (chained, single command buffer), CPU for eigendecomposition (Jacobi ≤ 128, Accelerate LAPACK > 128).

| shape | metal(ms) | sklearn(ms) | speedup |
|---|---|---|---|
| tall N=100K D=128 K=32 | 65 | 49 | 0.7× |
| sq N=1K D=512 K=32 | **22** | 55 | **2.5×** |
| sq N=5K D=1K K=32 | 144 | 184 | 1.3× |
| wide N=500 D=4K K=32 | 94 | 85 | 0.9× |
| wide N=500 D=8K K=32 | **176** | 219 | **1.2×** |
| wide N=500 D=16K K=32 | 334 | 327 | 1.0× |

GPU matches or beats sklearn in square and wide regimes; tall regimes are close (0.7×) due to GPU command buffer overhead (~2 ms) dominating a cheap 128×128 Gram solve. On medium-square shapes, GPU is **2.5× faster** than sklearn by avoiding SVD overhead on the Gram path.

### Linear Regression fit benchmarks (Apple M3)

```sh
cargo bench --bench linear_regression_benchmark
```

| N | D | Metal | CPU | sklearn | Speedup |
|---|---|---|---|---|---|
| 10K | 8 | 1.75 ms | 0.53 ms | 1.95 ms | 1.1× vs sklearn |
| 10K | 32 | 2.75 ms | 3.90 ms | 4.33 ms | 1.4× vs CPU |
| 100K | 8 | 4.84 ms | 5.29 ms | 9.13 ms | 1.1× vs CPU |
| 100K | 32 | 8.89 ms | 38.93 ms | 31.50 ms | **4.4× vs CPU** |
| 100K | 128 | 49.97 ms | 227.72 ms | 238.83 ms | **4.6× vs CPU** |
| 1M | 8 | 15.72 ms | 52.31 ms | 72.71 ms | **3.3× vs CPU** |
| 1M | 32 | 67.73 ms | 389.85 ms | 412.52 ms | **5.8× vs CPU** |

The Gram build streams the dataset with ideal `n·d` memory traffic: shared-memory
row tiles for D ≤ 128 (tiled kernel), chunked element columns for Xᵀy/colsum,
plus a fixed-order reduction. Tiny shapes (10K×8) are latency-bound (~2 ms
command-buffer overhead), but the GPU wins from 100K rows / D ≥ 32 up to
**5.8× vs CPU and 6.1× vs sklearn** at 1M×32.

### PCA kernel dispatch

- **Cov path** (N ≥ D): `centered^T @ centered / N` → D×D Gram → CPU eigh
- **Gram path** (N < D): `centered @ centered^T / N` → N×N Gram → CPU eigh + eigenvector recovery (`X^T @ U · diag(1/√(Nλ))`)
- Hybrid eigh: Jacobi (15 sweeps) for gram_dim ≤ 128, Accelerate `ssyevd_` for larger

## Kernel dispatch

| Condition | Kernel | Threads/TG | Points/TG |
|---|---|---|---|
| D < 8 \|\| D % 8 != 0 | `kmeans_assign` (Naive) | 256 | 1 |
| K ≤ 16, shared memory fits | `kmeans_assign_simdgroup_c16` (CTILE=16) | 128 | 8 |
| Shared memory fits | `kmeans_assign_simdgroup` (CTILE=8) | 128 | 8 |
| D > 0, shared memory exceeded | `kmeans_assign_splitd` | 128 | 128 |

Centroid update: GPU (`kmeans_centroid_tiled`) when `(K×D+K)×4 ≤ 32 KB`, else CPU fallback.

## Project structure

```
src/
├── lib.rs               – crate root + PyO3 pymodule entry
├── python.rs             – PyO3 bindings (KMeans, KNN, PCA, LogisticRegression, LinearRegression, GaussianNB)
├── metal/mod.rs          – MetalContext: device, queue, buffer helpers
├── kmeans/mod.rs         – KMeans, assign kernel picker, centroid dispatch
├── knn/mod.rs            – KNN, 3 kernel variants, buffer reuse
├── pca/mod.rs            – PCA: GPU Gram matrix + CPU eigh (Jacobi / Accelerate)
├── logistic_regression/mod.rs – LogisticRegression: fused fwd/bwd/loss kernels + L-BFGS
├── linear_regression/mod.rs   – LinearRegression: tiled Gram + chunked Xᵀy + host solve
└── naive_bayes/mod.rs    – GaussianNB: per-class sum/sumsq reduction + log-posterior predict kernel
python/
├── metal_kmeans/__init__.py  – KMeans + KNN Python API
├── metal_pca/__init__.py     – PCA Python API
├── metal_logistic_regression/__init__.py – LogisticRegression Python API
├── metal_linear_regression/__init__.py   – LinearRegression Python API
└── metal_gaussian_nb/__init__.py         – GaussianNB Python API
shaders/
├── kmeans.metal          – 5 Metal kernels (3 assign, 1 init, 1 centroid tiled)
├── knn.metal             – 4 Metal kernels (3 assign, 1 gather)
├── pca.metal             – 6 Metal kernels (mean, center, transpose, matmul, transform)
├── logistic.metal        – 4 Metal kernels (3 fused fwd/bwd, 1 predict)
├── linear.metal          – 5 Metal kernels (2 XᵀX, 1 Xᵀy, 1 reduce, 1 predict)
└── naive_bayes.metal     – 3 Metal kernels (reduce partials, reduce sum, predict logp)
examples/
├── example.py            – KMeans smoke test + benchmark
├── knn_example.py        – KNN benchmark across shapes
├── movie_recommendation.py – KNN movie recommendation engine
├── customer_segmentation.py – KMeans customer segmentation (500K rows)
├── pca_eigenfaces.py     – PCA eigenfaces reconstruction + face recognition
├── logistic_regression_example.py – LogisticRegression smoke test + benchmark
├── linear_regression_example.py   – LinearRegression smoke test + benchmark
└── diabetes_regression.py         – LinearRegression on real diabetes data (442×10)
docs/
├── algorithm.md          – algorithm deep-dive
├── python_api.md         – Python API reference
└── rust_api.md           – Rust API reference
```

## Notes

Apple GPU bugs worked around in `shaders/kmeans.metal`:

1. `simdgroup_load(base, stride, col, row)` produces garbage when `col ≠ 0` or `row ≠ 0`. Fix: adjust base pointer instead.
2. `simdgroup_multiply(acc, A, B)` **replaces** `acc` with `A*B` rather than accumulating. Fix: store each dim-tile to a separate shared-memory slot and sum via threads.
3. The Metal implementation uses row-first addressing (`pointer[(row+i)*stride + (col+j)]`) despite the spec describing column-first.
