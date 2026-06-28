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

## Feature flags

| Feature | Enables | Default |
|---|---|---|
| `python` | PyO3 bindings (`#[pymodule]` + `python.rs`) | Off |

Without `python`, the crate builds as a pure Rust library with no Python dependencies.

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
