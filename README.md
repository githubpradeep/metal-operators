# metal-kmeans

GPU-accelerated **KMeans clustering** and **K-Nearest Neighbors** via Apple Metal.

**KMeans** uses 5 kernel variants (simdgroup, split-D, tiled centroid) to run Lloyd's
algorithm entirely on GPU — no CPU readback inside the loop.

**KNN** uses 3 kernel variants (Dense direct-read for D<32, simdgroup Splitm for
D≥32&D%8=0, Naive fallback) with buffer reuse for 2.6–22× speedup over BLAS-CPU.

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

## Tests

```sh
cargo test                     # Rust: 25+ integration tests
python3 examples/example.py    # Python KMeans smoke test
python3 examples/knn_example.py # Python KNN smoke test
```

KMeans test matrix: D = {2, 4, 8, 16, 32, 64, 128}, K = {1, 8, 16, 32, 33, 64, 256}, including adjusted Rand index validation against CPU reference, multi-simdgroup correctness, split-D, empty-cluster handling, and timing.

KNN test matrix: D = {3, 8, 16, 32}, K = {1, 3, 5, 10}, covering Dense, Splitm, and Naive kernel paths plus deterministic reproducibility.

## Benchmarks

```sh
cargo bench --bench kmeans_benchmark
cargo bench --bench kmeans_benchmark -- "fit.*1M_D=32"
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
├── python.rs             – PyO3 bindings (MetalKMeans, MetalKNeighbors, functions)
├── metal/mod.rs          – MetalContext: device, queue, buffer helpers
├── kmeans/mod.rs         – KMeans, assign kernel picker, centroid dispatch
└── knn/mod.rs            – KNN, 3 kernel variants, buffer reuse
python/metal_kmeans/
├── __init__.py           – flashlib-style Python API (KMeans + KNN)
└── _native.pyi           – type stubs
shaders/
├── kmeans.metal          – 5 Metal kernels (3 assign, 1 init, 1 centroid tiled)
└── knn.metal             – 4 Metal kernels (3 assign, 1 gather)
examples/
├── example.py            – KMeans smoke test + benchmark
├── knn_example.py        – KNN benchmark across shapes
├── movie_recommendation.py – KNN movie recommendation engine
└── customer_segmentation.py – KMeans customer segmentation (500K rows)
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
