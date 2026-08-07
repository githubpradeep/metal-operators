# Rust API Reference

Core crate `metal-operators` re-exported as `metal_operators` for Rust consumers.

```toml
[dependencies]
metal-operators = "0.1"

# optional Python bindings
# metal-operators = { version = "0.1", features = ["python"] }
```

---

## `metal::MetalContext`

Wraps the Metal `Device` and `CommandQueue`.

```rust
pub struct MetalContext {
    pub device: Device,    // metal::Device
    pub queue: CommandQueue, // metal::CommandQueue
}
```

### `MetalContext::new() -> anyhow::Result<Self>`

Opens the system-default Metal device and creates a command queue. Fails with `"No Metal device found"` on non-Apple hardware.

### `compile_kernel(source: &str, name: &str) -> anyhow::Result<ComputePipelineState>`

Compiles a Metal Shading Language source string and returns a compute pipeline for function `name`. Metal functions are compiled JIT; results should be cached by the caller.

Errors are propagated from the Metal compiler (syntax errors, missing functions).

### `new_buffer<T>(data: &[T]) -> Buffer`

Creates a `StorageModeShared` Metal buffer initialized from the slice. Both CPU and GPU can access it without explicit synchronisation.

### `new_buffer_uninitialized(byte_size: u64) -> Buffer`

Allocates a `StorageModeShared` buffer of `byte_size` bytes. Contents are undefined.

### `read_buffer<T>(&self, buffer: &Buffer, count: usize) -> Vec<T>`

Copies `count` elements of type `T` from the Metal buffer into a new `Vec<T>`. Uses an unsafe pointer copy under the hood.

---

## `kmeans::KMeansConfig`

Configuration for the KMeans solver.

```rust
pub struct KMeansConfig {
    pub k: usize,                    // number of clusters
    pub max_iterations: usize,       // Lloyd iteration limit
    pub tolerance: f32,              // convergence threshold (max centroid shift)
    pub seed: u64,                   // RNG seed (0 = time-based)
    pub init_centroids: Option<Vec<f32>>,  // optional pre-defined centroids
}
```

Implements `Default`:

| Field | Default |
|---|---|
| `k` | `8` |
| `max_iterations` | `100` |
| `tolerance` | `1e-4` |
| `seed` | `42` |
| `init_centroids` | `None` |

---

## `kmeans::KMeans`

The main solver struct. All fields are private.

```rust
pub struct KMeans { /* private fields */ }
```

### `KMeans::new(config: KMeansConfig) -> Self`

Construct a new solver. No GPU work is performed until `fit` is called.

### `fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()>`

Run Lloyd's algorithm.

| Param | Type | Description |
|---|---|---|
| `ctx` | `&MetalContext` | GPU handle (shared across calls). |
| `data` | `&[f32]` | Flat row-major points: `data[i * d + j]` = point `i`, dim `j`. Length `n * d`. |
| `n` | `usize` | Number of points. |
| `d` | `usize` | Number of dimensions. |

**Validates**:
- `d > 0`, `n > 0`, `k > 0`, `k <= n`
- `data.len() == n * d`
- If `init_centroids` is `Some`, its length must be `k * d`

**Algorithm**:
1. k-means++ initialization (GPU, unless `init_centroids` is provided).
2. For each iteration:
   - Dispatch assign kernel (auto-selected: Naive / Simdgroup / SimdgroupC16 / SplitD).
   - Dispatch `kmeans_centroid_tiled` on GPU when shared memory budget permits; otherwise read assignments to CPU and compute centroids on CPU.
   - Check convergence via `max_centroid_shift`.
3. Read labels back, compute inertia.

### `predict(&self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<Vec<usize>>`

Assign points to the **existing** fitted centroids. Does not mutate `self`.

| Param | Type | Description |
|---|---|---|
| `ctx` | `&MetalContext` | GPU handle. |
| `data` | `&[f32]` | Flat row-major points. |
| `n` | `usize` | Number of points. |
| `d` | `usize` | Number of dimensions. |

Returns a `Vec<usize>` of length `n` with cluster labels 0..k-1.

**Requires**: `fit` must have been called first (centroids must exist).

### Accessors

| Method | Returns | Description |
|---|---|---|
| `centroids(&self) -> &[f32]` | `&[f32]` | Flat centroids, length `k * d`. |
| `labels(&self) -> &[usize]` | `&[usize]` | Per-point labels from last fit. |
| `inertia(&self) -> f32` | `f32` | Within-cluster sum of squared distances. |
| `n_iter(&self) -> usize` | `usize` | Iterations used on last fit. |

---

## Internal dispatch

### Assign kernel picker (`pick_assign_kernel`)

| Condition | Kernel | Threads/TG | Points/TG |
|---|---|---|---|
| `d < 8` or `d % 8 != 0` | `kmeans_assign` (Naive) | 256 | 1 |
| `k <= 16` and shared ≤ 32 KB | `kmeans_assign_simdgroup_c16` | 128 | 8 |
| Shared ≤ 32 KB | `kmeans_assign_simdgroup` | 128 | 8 |
| Otherwise | `kmeans_assign_splitd` | 128 | 128 |

### Centroid update

| Condition | Method |
|---|---|
| `(k * d + k) * 4 <= 32_768` | `kmeans_centroid_tiled` (GPU) |
| Shared memory exceeded | CPU `compute_centroids` fallback |

---

## `pca::PCAConfig`

Configuration for PCA.

```rust
pub struct PCAConfig {
    pub n_components: usize,
}
```

Implements `Default`:

| Field | Default |
|---|---|
| `n_components` | `2` |

---

## `pca::PCA`

The main PCA solver struct.

```rust
pub struct PCA { /* private fields */ }
```

### `PCA::new(config: PCAConfig) -> Self`

Construct a new PCA solver. No GPU work is performed until `fit` is called.

### `fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()>`

Fit PCA on the GPU.

| Param | Type | Description |
|---|---|---|
| `ctx` | `&MetalContext` | GPU handle (shared across calls). |
| `data` | `&[f32]` | Flat row-major data: `data[i * d + j]` = sample `i`, dim `j`. Length `n * d`. |
| `n` | `usize` | Number of samples. |
| `d` | `usize` | Number of features. |

**Algorithm**:
1. GPU pipeline (single command buffer): `mean → center → transpose → matmul`.
2. Read Gram matrix and means back to CPU.
3. CPU eigendecomposition — Jacobi (≤ 128) or Accelerate `ssyevd_` (> 128).
4. Sort eigenvalues descending, extract top-K components.
5. If Gram path (N < D), recover D-dim eigenvectors via `X̂ᵀ @ U · diag(1/√(Nλ))`.

### `transform(&self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<Vec<f32>>`

Project data onto principal components (GPU).

| Param | Type | Description |
|---|---|---|
| `ctx` | `&MetalContext` | GPU handle. |
| `data` | `&[f32]` | Flat row-major data, length `n * d`. |
| `n` | `usize` | Number of samples. |
| `d` | `usize` | Number of features. |

Returns flat `Vec<f32>` of length `n * k` where `k` = requested components.

### `fit_transform(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<Vec<f32>>`

Fit + transform in one call.

### Accessors

| Method | Returns | Description |
|---|---|---|
| `components(&self) -> &[f32]` | `&[f32]` | Flat principal components, length `k * d`, row-major. |
| `explained_variance(&self) -> &[f32]` | `&[f32]` | Variance of each component (descending), length `k`. |
| `explained_variance_ratio(&self) -> &[f32]` | `&[f32]` | Normalized variance (sums to ≤ 1), length `k`. |
| `mean(&self) -> &[f32]` | `&[f32]` | Per-feature mean, length `d`. |
| `singular_values(&self) -> &[f32]` | `&[f32]` | Singular values `sqrt(N · λ)`, length `k`. |
| `noise_variance(&self) -> f32` | `f32` | Average variance of discarded components. |
| `n_features(&self) -> usize` | `usize` | Feature dimension `d`. |
| `n_samples(&self) -> usize` | `usize` | Sample count `n`. |

### Example

```rust ignore
use metal_operators::pca::{PCA, PCAConfig};
use metal_operators::metal::MetalContext;

fn run() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;

    let (n, d, k) = (1000usize, 50, 5);
    let data = vec![0.0f32; n * d];  // your data here

    let mut pca = PCA::new(PCAConfig { n_components: k });
    pca.fit(&ctx, &data, n, d)?;

    println!("components: {:?}", &pca.components()[..k * d.min(4)]);
    println!("explained variance: {:?}", &pca.explained_variance()[..k.min(5)]);

    let transformed = pca.transform(&ctx, &data, n, d)?;
    println!("transformed: {} × {}", n, k);
    Ok(())
}
```

---

## Feature flags

| Feature | Enables | Default |
|---|---|---|
| `python` | PyO3 bindings (`#[pymodule]` + `python.rs`) | Off |

Without `python`, the crate builds as a pure Rust library with no Python dependencies.

---

## `knn::KNNConfig`

Configuration for K-Nearest Neighbors search.

```rust
pub struct KNNConfig {
    pub k: usize,  // number of neighbours
}
```

Implements `Default`:

| Field | Default |
|---|---|
| `k` | `5` |

---

## `knn::KNN`

The main KNN struct. All fields are private.

```rust
pub struct KNN { /* private fields */ }
```

### `KNN::new(config: KNNConfig) -> Self`

Construct a new KNN searcher. No GPU work is performed until `fit` is called.

### `fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()>`

Store the corpus (database) on the GPU and select the optimal kernel variant.

| Param | Type | Description |
|---|---|---|
| `ctx` | `&MetalContext` | GPU handle (shared across calls). |
| `data` | `&[f32]` | Flat row-major corpus: `data[i * d + j]` = point `i`, dim `j`. Length `n * d`. |
| `n` | `usize` | Number of corpus points. |
| `d` | `usize` | Number of dimensions. |

**Validates**:
- `d > 0`, `n > 0`
- `data.len() == n * d`

**Kernel selection** (based on `d` and `k`):

| Condition | Kernel | Description |
|---|---|---|
| `d < 32` and `k ≤ 64` | `knn_assign_dense` | Direct device reads, register-resident query, per-thread heap |
| `d ≥ 8`, `d % 8 == 0`, `k ≤ 64` | `knn_assign_splitm` | Simdgroup matmul (BN=16, BM=8), shared memory tiling |
| Otherwise | `knn_assign_naive` | Single-thread fallback |

The corpus and its pre-computed squared norms are uploaded to GPU memory during
`fit` and stay resident for all subsequent `kneighbors` calls.

### `kneighbors(&self, ctx: &MetalContext, queries: &[f32], nq: usize) -> anyhow::Result<(Vec<f32>, Vec<u32>)>`

Find the `k` nearest neighbours of each query point.

| Param | Type | Description |
|---|---|---|
| `ctx` | `&MetalContext` | GPU handle (shared across calls). |
| `queries` | `&[f32]` | Flat row-major queries, length `nq * d`. |
| `nq` | `usize` | Number of query points. |

Returns `(distances, indices)`:

| Return | Type | Length | Description |
|---|---|---|---|
| `distances` | `Vec<f32>` | `nq * k` | Squared Euclidean distances, row-major, sorted ascending per query. |
| `indices` | `Vec<u32>` | `nq * k` | Corpus indices of neighbours (0..n-1). |

**Internal flow**:
1. Compute query squared norms on CPU (`compute_norms`).
2. Reuse cached scratch buffers for query data, norms, and output arrays.
3. Dispatch the selected GPU kernel (single threadgroup grid, no M-split).
4. Read GPU output back to CPU.
5. Add query norms to recover true squared-L2 from the shift-invariant score.

**Buffer reuse**: query, norms, and output Metal buffers are cached across calls.
Only a `copy_nonoverlapping` CPU→GPU upload of query data occurs on each
invocation — no `new_buffer` allocation in the hot path.

### Example

```rust ignore
use metal_operators::knn::{KNN, KNNConfig};
use metal_operators::metal::MetalContext;

fn run() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;
    let (nc, nq, d, k) = (10_000, 1_000, 32, 5);

    let corpus = vec![0.0f32; nc * d];
    let queries = vec![0.0f32; nq * d];

    let mut knn = KNN::new(KNNConfig { k });
    knn.fit(&ctx, &corpus, nc, d)?;
    let (distances, indices) = knn.kneighbors(&ctx, &queries, nq)?;

    println!("distances: {:?}", &distances[..k]);
    println!("indices:   {:?}", &indices[..k]);
    Ok(())
}
```

---

## `dbscan::DBSCANConfig`

```rust ignore
pub struct DBSCANConfig {
    pub eps: f32,
    pub min_samples: usize,
}
```

| Field | Default | Meaning |
|---|---|---|
| `eps` | `0.5` | Radius of the ε-neighborhood (must be `> 0`). |
| `min_samples` | `5` | Minimum number of points (including the point itself) within `eps` for a point to be a core point (must be `>= 1`). |

## `dbscan::DBSCAN`

DBSCAN density-based clustering. The expensive ε-neighborhood computation is
done on the GPU with the `dbscan_count` / `dbscan_gather` kernels
(`shaders/dbscan.metal`), which reuse the same squared-L2 distance approach as
the KNN distance kernels (`shaders/knn.metal`) but enumerate every neighbor
within `eps` (no per-query `k` cap, so connectivity is exact). Core-point
detection, mutual-core union-find and border/noise labelling run on the CPU.

Labels follow scikit-learn conventions: `-1` = noise, `0..n_clusters-1` =
cluster ids assigned in first-appearance order.

```rust ignore
pub struct DBSCAN { /* private fields */ }
```

### `DBSCAN::new(config: DBSCANConfig) -> Self`

Construct a new DBSCAN clusterer. No GPU work is performed until `fit`.

### `fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()>`

Fits the model:

1. Validates `n > 0`, `d > 0`, `eps > 0`, `min_samples >= 1` and
   `data.len() == n * d`.
2. GPU pass A (`dbscan_count`): counts, for each point, the number of *other*
   points within `eps`.
3. CPU prefix-sum over the counts to build CSR offsets; GPU pass B
   (`dbscan_gather`) fills the compact neighbor lists.
4. Core points: `count + 1 (self) >= min_samples`.
5. Union-find over core points connected by ε-adjacency (exact graph), then
   border points join the first core neighbor's cluster.

### Accessors

```rust ignore
impl DBSCAN {
    pub fn labels(&self) -> &[isize];     // cluster id or -1 (noise)
    pub fn n_clusters(&self) -> usize;    // number of clusters found
}
```

### Example

```rust ignore
use metal_operators::dbscan::{DBSCAN, DBSCANConfig};
use metal_operators::metal::MetalContext;

fn run() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;
    let (n, d) = (100, 2);
    let data = vec![0.0f32; n * d];

    let mut db = DBSCAN::new(DBSCANConfig { eps: 0.5, min_samples: 5 });
    db.fit(&ctx, &data, n, d)?;
    println!("clusters: {}", db.n_clusters());
    println!("labels:   {:?}", db.labels());
    Ok(())
}
```

---

## `tsne::TSNEConfig`

```rust ignore
pub struct TSNEConfig {
    pub n_components: usize,
    pub perplexity: f32,
    pub learning_rate: f32,
    pub n_iter: usize,
    pub early_exaggeration: f32,
    pub exaggeration_iter: usize,
    pub momentum: f32,
    pub seed: u64,
    pub min_grad_norm: f32,
}
```

| Field | Default | Meaning |
|---|---|---|
| `n_components` | `2` | Embedding dimensions (1..=8; the GPU kernel loops over this bounded range). |
| `perplexity` | `30.0` | Target perplexity; must satisfy `1 <= perplexity < n_samples`. |
| `learning_rate` | `200.0` | Gradient-descent step size (must be `> 0`). |
| `n_iter` | `1000` | Number of gradient descent iterations. |
| `early_exaggeration` | `12.0` | Multiplier applied to `P` for the first `exaggeration_iter` iterations (`>= 1.0`; `1.0` disables). |
| `exaggeration_iter` | `250` | Number of early-exaggeration iterations. |
| `momentum` | `0.8` | Velocity (momentum) coefficient. |
| `seed` | `42` | Seed for the Gaussian (`0, 1e-4`) embedding initialization. |
| `min_grad_norm` | `1e-7` | Early-stopping threshold on the gradient L2 norm; `0.0` disables. |

## `tsne::TSNE`

t-SNE nonlinear dimensionality reduction / embedding (scikit-learn `exact`
semantics: `P = (P_{j|i} + P_{i|j}) / (2N)` with a zero diagonal, early
exaggeration of `P`, random Gaussian init, and the classic velocity momentum
update `v ← momentum·v − lr·grad; Y ← Y + v`).

The two dominant exact-t-SNE costs are GPU-accelerated in `shaders/tsne.metal`:

1. **Pairwise affinity `P` (O(N²·D), once).** `tsne_distances` computes the full
   squared-L2 distance matrix; `tsne_perplexity` runs the per-row perplexity
   bisection over `log(σ)` in parallel (one thread per sample, an
   embarrassingly-parallel workload).
2. **Gradient (O(N²), per iteration).** `tsne_grad` computes every sample's
   gradient in parallel — one thread per point, so there are no intra-kernel
   races. The `÷Z` Student-t normalization is folded in on the host, and the
   O(N) momentum update runs on the CPU.

```rust ignore
pub struct TSNE { /* private fields */ }
```

### `TSNE::new(config: TSNEConfig) -> Self`

Construct a new t-SNE embedding operator. No GPU work is performed until `fit`.

### `fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()>`

Fits the embedding:

1. Validates `n > 1`, `d > 0`, `1 <= n_components <= 8`,
   `1 <= perplexity < n`, `learning_rate > 0`, `early_exaggeration >= 1.0` and
   `data.len() == n * d`.
2. GPU `tsne_distances` → full N×N distance matrix.
3. GPU `tsne_perplexity` → conditional `P_{j|i}`, then symmetrized/normalized
   on the host: `P = (P_{j|i} + P_{i|j}) / (2N)`.
4. Random Gaussian embedding init (seeded; deterministic), then
   `n_iter` gradient iterations. Early stopping triggers when the gradient L2
   norm drops below `min_grad_norm`.
5. Final KL(P‖Q) is computed on the host for diagnostics.

### Accessors

```rust ignore
impl TSNE {
    pub fn embedding(&self) -> &[f32];    // (n, n_components) row-major
    pub fn n_iter(&self) -> usize;        // iterations actually run
    pub fn kl_divergence(&self) -> f32;   // final KL(P ‖ Q)
}
```

### Example

```rust ignore
use metal_operators::metal::MetalContext;
use metal_operators::tsne::{TSNE, TSNEConfig};

fn run() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;
    let (n, d) = (200, 8);
    let data = vec![0.0f32; n * d];

    let mut tsne = TSNE::new(TSNEConfig {
        n_components: 2,
        perplexity: 30.0,
        n_iter: 500,
        ..Default::default()
    });
    tsne.fit(&ctx, &data, n, d)?;
    println!("embedding: {}×{}", n, tsne.embedding().len() / n);
    println!("KL(P‖Q):   {}", tsne.kl_divergence());
    Ok(())
}
```

---

## `nmf::NMFConfig`

Configuration for Non-negative Matrix Factorization (multiplicative updates).

```rust
pub struct NMFConfig {
    pub n_components: usize,     // latent rank K (clamped to input n/d)
    pub max_iterations: usize,   // multiplicative-update iteration limit
    pub tolerance: f32,          // early stop: relative Frobenius change of W
    pub seed: u64,               // RNG seed for the non-negative init
    pub eps: f32,                // guard against division by zero
}
```

Implements `Default`:

| Field | Default |
|---|---|
| `n_components` | `2` |
| `max_iterations` | `200` |
| `tolerance` | `1e-4` |
| `seed` | `42` |
| `eps` | `1e-10` |

---

## `nmf::NMF`

Non-negative matrix factorization: `V (N×D) ≈ W·H` with `W (N×K)` and
`H (K×D)` both ≥ 0. Every O(N·D·K) step is a Metal matmul (see
`shaders/nmf.metal`): a generic `nmf_mm` kernel (with a per-operand
transpose flag) computes `Wᵀ·V`, `Wᵀ·W`, `Wᵀ·W·H`, `V·Hᵀ`, `H·Hᵀ` and
`W·H·Hᵀ`, while `nmf_update` applies the Lee–Seung multiplicative rule in
place and `nmf_diff` computes the reconstruction residuals.

```rust
pub struct NMF { /* private fields */ }
```

### `NMF::new(config: NMFConfig) -> Self`

Construct a new solver. No GPU work is performed until `fit` is called.

### `fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()>`

Run the Lee–Seung multiplicative updates.

| Param | Desc |
|---|---|
| `data` | Flat row-major, **non-negative** matrix: `data[i * d + j]`. Length `n * d`. |
| `n` | Number of rows (samples). |
| `d` | Number of columns (features). |

Validates `n > 0`, `d > 0`, `data.len() == n * d`, and that every entry is
non-negative.

### `transform(&mut self, ctx, data: &[f32], n, d) -> anyhow::Result<Vec<f32>>`

Project new (non-negative) data into the latent space: solve `X ≈ W_new·H`
with the fitted components `H` fixed (multiplicative updates on `W_new`).
Returns the non-negative coefficient matrix `W_new`, length `n * k`.

### Accessors

- `components() -> &[f32]` — fitted basis `H` (K×D).
- `coeff() -> &[f32]` — coefficient matrix `W` (N×K) of the training data.
- `reconstruction_error() -> f32` — `‖V − W·H‖_F`.
- `n_iter() -> usize` — iterations run by the last `fit`.
- `n_samples() -> usize`, `n_features() -> usize`.

### Example

```rust ignore
use metal_operators::metal::MetalContext;
use metal_operators::nmf::{NMF, NMFConfig};

fn run() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;
    let (n, d, k) = (1000usize, 200usize, 8usize);
    let v = vec![1.0f32; n * d]; // non-negative data

    let mut nmf = NMF::new(NMFConfig {
        n_components: k,
        max_iterations: 200,
        tolerance: 1e-4,
        seed: 42,
        eps: 1e-10,
    });
    nmf.fit(&ctx, &v, n, d)?;
    println!("recon error: {:.4}", nmf.reconstruction_error());
    Ok(())
}
```

---

## Complete example

```rust ignore
use metal_operators::kmeans::{KMeans, KMeansConfig};
use metal_operators::metal::MetalContext;

fn run() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;

    // 100K points, 32 dimensions, 64 clusters
    let (n, d, k) = (100_000usize, 32, 64);
    let points = vec![0.0f32; n * d];  // your data here

    let mut km = KMeans::new(KMeansConfig {
        k,
        max_iterations: 15,
        tolerance: 1e-4,
        seed: 42,
        init_centroids: None,
    });

    km.fit(&ctx, &points, n, d)?;
    println!("inertia={}  n_iter={}", km.inertia(), km.n_iter());

    let labels = km.predict(&ctx, &points, n, d)?;
    println!("labels: {}..{}", labels[0], labels[n - 1]);
    Ok(())
}
```

---

## `gmm::GMMConfig`

Configuration for the Gaussian Mixture Model (full covariance, EM).

```rust
pub struct GMMConfig {
    pub n_components: usize,   // number of mixture components (clusters)
    pub max_iterations: usize, // EM iteration limit
    pub tolerance: f32,        // convergence: |Δ avg log-likelihood| < tol
    pub seed: u64,             // seed for the k-means++ init PRNG
    pub reg_covar: f32,        // diagonal regularization for covariance (>= 0)
}
```

Implements `Default`:

| Field | Default |
|---|---|
| `n_components` | `3` |
| `max_iterations` | `100` |
| `tolerance` | `1e-3` |
| `seed` | `42` |
| `reg_covar` | `1e-6` |

---

## `gmm::GMM`

Gaussian Mixture Model fitted with GPU-accelerated Expectation-Maximization,
mirroring `sklearn.mixture.GaussianMixture` (full covariance, EM loop,
k-means++ init). See `shaders/gmm.metal`.

The EM iteration structure:

1. **Init.** k-means++ selects `n_components` seed means; each component's
   covariance starts as the empirical covariance of its nearest points
   (falling back to the global covariance for empty/small clusters), plus
   `reg_covar` on the diagonal.
2. **E-step (GPU).** `gmm_e_step` computes the (n × k) log-likelihood matrix
   `L[i][c] = log w_c − ½(d log 2π + log|Σ_c| + (xᵢ−μ_c)ᵀ Σ_c⁻¹ (xᵢ−μ_c))`
   in a single launch. The host converts it to responsibilities
   `r_ic = softmax_c(L_i)` and the average log-likelihood lower bound.
3. **M-step (host).** Weights, means, and full covariances are re-estimated
   from the responsibilities. A per-component Cholesky factorization rebuilds
   the precision `Σ_c⁻¹` and `log|Σ_c|` for the next E-step (degenerate
   covariances get progressively more diagonal regularization).
4. **Loop** until `|lower_bound − previous| < tolerance` or `max_iterations`.

```rust ignore
pub struct GMM { /* private fields */ }
```

### `GMM::new(config: GMMConfig) -> Self`

Construct a new Gaussian Mixture Model. No GPU work is performed until `fit`.

### `fit(&mut self, ctx: &MetalContext, data: &[f32], n: usize, d: usize) -> anyhow::Result<()>`

Fits the mixture by EM:

1. Validates `n > 0`, `d > 0`, `1 <= n_components <= n`, `tolerance >= 0`,
   `reg_covar >= 0`, `max_iterations > 0`, and `data.len() == n * d`.
2. k-means++ init (seeded; deterministic).
3. Iterates E-step (GPU) / M-step (host) until convergence.

### `predict`, `predict_proba`, `score`

Each is a single GPU launch against the fitted parameters:

```rust ignore
impl GMM {
    pub fn predict(&self, ctx: &MetalContext, data: &[f32], n: usize, d: usize)
        -> anyhow::Result<Vec<usize>>; // hard component assignment (n,)
    pub fn predict_proba(&self, ctx: &MetalContext, data: &[f32], n: usize, d: usize)
        -> anyhow::Result<Vec<f32>>;    // responsibilities (n × k), rows sum to 1
    pub fn score(&self, ctx: &MetalContext, data: &[f32], n: usize, d: usize)
        -> anyhow::Result<f32>;         // average log-likelihood
}
```

### Accessors

```rust ignore
impl GMM {
    pub fn n_components(&self) -> usize;      // k
    pub fn n_features(&self) -> usize;        // d
    pub fn n_samples(&self) -> usize;         // n
    pub fn weights(&self) -> &[f32];          // (k,)
    pub fn means(&self) -> &[f32];            // (k, d) row-major
    pub fn covariances(&self) -> &[f32];      // (k, d, d) row-major
    pub fn precisions(&self) -> &[f32];       // (k, d, d) Σ⁻¹
    pub fn responsibilities(&self) -> &[f32]; // (n, k) from last fit
    pub fn lower_bound(&self) -> f32;         // avg log-likelihood lower bound
    pub fn n_iter(&self) -> usize;            // EM iterations actually run
}
```

### Example

```rust ignore
use metal_operators::metal::MetalContext;
use metal_operators::gmm::{GMM, GMMConfig};

fn run() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;
    let (n, d) = (600, 3);
    let data = vec![0.0f32; n * d]; // three well-separated blobs

    let mut gmm = GMM::new(GMMConfig {
        n_components: 3,
        max_iterations: 100,
        tolerance: 1e-4,
        seed: 42,
        reg_covar: 1e-4,
    });
    gmm.fit(&ctx, &data, n, d)?;
    println!("lower bound: {}", gmm.lower_bound());
    println!("iterations:  {}", gmm.n_iter());
    Ok(())
}
```
