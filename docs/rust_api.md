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
